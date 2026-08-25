from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .aggregation import masked_mean_aggregate
from .graph_layers import LaplacianPredictor, faces_to_edge_index
from .image_encoder import ImageFeatureConstructor, SmallImageEncoder
from .projection import project_vertices, sample_vertex_features
from .query_training import QUERY_FOURIER_GEOMETRY_MODE
from .target_scaling import mean_incident_edge_length


INPUT_MODES = {"coarse_only", "multiview_only", "coarse_plus_multiview"}
GEOMETRY_MODES = {"legacy", QUERY_FOURIER_GEOMETRY_MODE}


class FourierPositionEncoding(nn.Module):
    """Fourier features for shape-normalized three-dimensional query positions."""

    def __init__(self, num_frequencies: int = 6, include_input: bool = True) -> None:
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative.")
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        frequencies = torch.pi * torch.pow(2.0, torch.arange(num_frequencies))
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return 3 * (2 * self.num_frequencies + int(self.include_input))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        phases = positions.float().unsqueeze(-1) * self.frequencies.float()
        encoded = [
            torch.sin(phases).flatten(start_dim=1),
            torch.cos(phases).flatten(start_dim=1),
        ]
        if self.include_input:
            encoded.insert(0, positions.float())
        return torch.cat(encoded, dim=-1)


@dataclass(frozen=True)
class LearnedLaplacianOutput:
    predicted_laplacian: torch.Tensor
    confidence_prediction: torch.Tensor | None
    vertex_features: torch.Tensor
    aggregated_image_features: torch.Tensor
    valid_view_ratio: torch.Tensor
    valid_views: torch.Tensor
    oracle_residual_prediction: torch.Tensor | None = None
    base_laplacian_prediction: torch.Tensor | None = None
    dynamic_expert_residual_prediction: torch.Tensor | None = None
    dynamic_gate_logit: torch.Tensor | None = None
    dynamic_gate_signed: torch.Tensor | None = None
    dynamic_gate_effective: torch.Tensor | None = None
    recovery_lambda_logit: torch.Tensor | None = None
    recovery_lambda: torch.Tensor | None = None
    direct_vertex_displacement_prediction: torch.Tensor | None = None

    @property
    def delta_hat_prediction(self) -> torch.Tensor:
        """Explicit canonical name for the absolute normalized model output."""

        return self.predicted_laplacian


class LearnedLaplacianModel(nn.Module):
    """CNN + differentiable projection + graph predictor for one prediction mesh."""

    GEOMETRY_DIM = 10  # position, normal, initial Laplacian, log(1 + degree)

    def __init__(
        self,
        image_feature_dim: int = 32,
        image_first_stride: int = 2,
        image_second_stride: int = 2,
        image_feature_construction_mode: str = "original",
        image_gaussian_kernel_size: int = 5,
        image_gaussian_sigma: float = 1.0,
        image_view_chunk_size: int | None = None,
        image_gradient_checkpointing: bool = False,
        hidden_dim: int = 128,
        num_graph_layers: int = 3,
        dropout: float = 0.0,
        input_mode: str = "coarse_plus_multiview",
        zero_images: bool = False,
        geometry_mode: str = "legacy",
        position_num_frequencies: int = 6,
        position_include_input: bool = True,
        predict_confidence: bool = False,
        oracle_residual_expert_enabled: bool = False,
        oracle_residual_expert_hidden_dim: int = 32,
        dynamic_residual_expert_enabled: bool = False,
        dynamic_residual_expert_hidden_dim: int = 32,
        dynamic_gate_hidden_dim: int = 32,
        dynamic_gate_initial_bias: float = 0.1,
        recovery_lambda_head_enabled: bool = False,
        recovery_lambda_head_hidden_dim: int = 16,
        recovery_lambda_minimum: float = 1e-3,
        recovery_lambda_maximum: float = 1e-1,
        recovery_lambda_initial: float = 1e-2,
        hybrid_direct_head_enabled: bool = False,
    ) -> None:
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}, got {input_mode!r}.")
        if geometry_mode not in GEOMETRY_MODES:
            raise ValueError(
                f"geometry_mode must be one of {sorted(GEOMETRY_MODES)}, got {geometry_mode!r}."
            )
        self.image_feature_dim = image_feature_dim
        self.image_first_stride = int(image_first_stride)
        self.image_second_stride = int(image_second_stride)
        self.image_feature_constructor = ImageFeatureConstructor(
            mode=image_feature_construction_mode,
            gaussian_kernel_size=image_gaussian_kernel_size,
            gaussian_sigma=image_gaussian_sigma,
        )
        self.projected_image_feature_dim = (
            image_feature_dim
            * self.image_feature_constructor.output_channel_multiplier
        )
        if image_view_chunk_size is not None and image_view_chunk_size < 1:
            raise ValueError("image_view_chunk_size must be positive when provided.")
        self.image_view_chunk_size = image_view_chunk_size
        self.image_gradient_checkpointing = bool(image_gradient_checkpointing)
        self.input_mode = input_mode
        self.zero_images = zero_images
        self.geometry_mode = geometry_mode
        self.predict_confidence = bool(predict_confidence)
        self.oracle_residual_expert_enabled = bool(oracle_residual_expert_enabled)
        self.oracle_residual_expert_hidden_dim = int(oracle_residual_expert_hidden_dim)
        if self.oracle_residual_expert_hidden_dim < 1:
            raise ValueError("oracle_residual_expert_hidden_dim must be positive.")
        self.dynamic_residual_expert_enabled = bool(dynamic_residual_expert_enabled)
        self.dynamic_residual_expert_hidden_dim = int(
            dynamic_residual_expert_hidden_dim
        )
        self.dynamic_gate_hidden_dim = int(dynamic_gate_hidden_dim)
        self.dynamic_gate_initial_bias = float(dynamic_gate_initial_bias)
        self.recovery_lambda_head_enabled = bool(recovery_lambda_head_enabled)
        self.recovery_lambda_head_hidden_dim = int(recovery_lambda_head_hidden_dim)
        self.recovery_lambda_minimum = float(recovery_lambda_minimum)
        self.recovery_lambda_maximum = float(recovery_lambda_maximum)
        self.recovery_lambda_initial = float(recovery_lambda_initial)
        self.hybrid_direct_head_enabled = bool(hybrid_direct_head_enabled)
        if self.dynamic_residual_expert_hidden_dim < 1:
            raise ValueError("dynamic_residual_expert_hidden_dim must be positive.")
        if self.dynamic_gate_hidden_dim < 1:
            raise ValueError("dynamic_gate_hidden_dim must be positive.")
        if self.recovery_lambda_head_hidden_dim < 1:
            raise ValueError("recovery_lambda_head_hidden_dim must be positive.")
        if not (
            0 < self.recovery_lambda_minimum
            < self.recovery_lambda_initial
            < self.recovery_lambda_maximum
        ):
            raise ValueError(
                "Recovery lambda bounds must satisfy 0 < minimum < initial < maximum."
            )
        self.position_encoder = FourierPositionEncoding(
            num_frequencies=position_num_frequencies,
            include_input=position_include_input,
        )
        self.image_encoder = SmallImageEncoder(
            image_feature_dim,
            first_stride=self.image_first_stride,
            second_stride=self.image_second_stride,
        )
        geometry_dim = (
            self.GEOMETRY_DIM
            if geometry_mode == "legacy"
            else self.position_encoder.output_dim + 3 + 1 + 1
        )
        self.predictor = LaplacianPredictor(
            input_dim=geometry_dim + 1 + self.projected_image_feature_dim,
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
        )
        # Confidence is deliberately a small side head.  The differential target
        # backbone and its three-vector output remain unchanged.
        self.confidence_head = (
            nn.Sequential(
                nn.Linear(
                    geometry_dim + 1 + self.projected_image_feature_dim,
                    hidden_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            if self.predict_confidence
            else None
        )
        # Construct the oracle diagnostic branch after every canonical module so
        # enabling it cannot change initialization of the general predictor or
        # confidence head. Its zero-initialized output also makes E0/E1 initial
        # predictions identical.
        self.oracle_residual_expert = (
            nn.Sequential(
                nn.Linear(hidden_dim, self.oracle_residual_expert_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.oracle_residual_expert_hidden_dim, 3),
            )
            if self.oracle_residual_expert_enabled
            else None
        )
        if self.oracle_residual_expert is not None:
            final = self.oracle_residual_expert[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        # The learned expert is deliberately constructed last so it cannot alter
        # the seeded initialization of the canonical MSE predictor.  A zero
        # residual output preserves the frozen base prediction exactly at step 0.
        # The positive, spatially uniform initial gate lets the residual output
        # layer receive gradients immediately without using target-derived routing.
        self.dynamic_residual_expert = (
            nn.Sequential(
                nn.Linear(hidden_dim, self.dynamic_residual_expert_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dynamic_residual_expert_hidden_dim, 3),
            )
            if self.dynamic_residual_expert_enabled
            else None
        )
        self.dynamic_gate_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, self.dynamic_gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dynamic_gate_hidden_dim, 1),
            )
            if self.dynamic_residual_expert_enabled
            else None
        )
        if self.dynamic_residual_expert is not None:
            residual_final = self.dynamic_residual_expert[-1]
            nn.init.zeros_(residual_final.weight)
            nn.init.zeros_(residual_final.bias)
            gate_final = self.dynamic_gate_head[-1]
            nn.init.zeros_(gate_final.weight)
            nn.init.constant_(gate_final.bias, self.dynamic_gate_initial_bias)
        # A mesh-level scalar side head. Its zero output-layer weights make the
        # initial lambda exactly the requested Arm-B operating point while
        # preserving every canonical per-vertex prediction.
        self.recovery_lambda_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, self.recovery_lambda_head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.recovery_lambda_head_hidden_dim, 1),
            )
            if self.recovery_lambda_head_enabled
            else None
        )
        if self.recovery_lambda_head is not None:
            final = self.recovery_lambda_head[-1]
            nn.init.zeros_(final.weight)
            low = torch.log10(torch.tensor(self.recovery_lambda_minimum)).item()
            high = torch.log10(torch.tensor(self.recovery_lambda_maximum)).item()
            initial = torch.log10(torch.tensor(self.recovery_lambda_initial)).item()
            fraction = (initial - low) / (high - low)
            bias = torch.logit(torch.tensor(fraction)).item()
            nn.init.constant_(final.bias, bias)
        # The controlled end-to-end hybrid adds exactly one output head to the
        # established shared backbone.  It mirrors the canonical Laplacian head
        # and is constructed last, so enabling it cannot perturb the seeded
        # initialization of any pre-existing parameter.
        self.hybrid_direct_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 3),
            )
            if self.hybrid_direct_head_enabled
            else None
        )

    def forward(self, sample: Mapping[str, Any]) -> LearnedLaplacianOutput:
        query_positions = sample.get("query_positions", sample["vertices"])
        if self.input_mode == "coarse_only":
            num_views = int(
                sample.get(
                    "num_views",
                    sample["intrinsics"].shape[0] if "intrinsics" in sample else 0,
                )
            )
            valid = torch.zeros(
                (num_views, query_positions.shape[0]),
                dtype=torch.bool,
                device=query_positions.device,
            )
            aggregated = query_positions.new_zeros(
                (query_positions.shape[0], self.projected_image_feature_dim)
            )
            valid_ratio = query_positions.new_zeros((query_positions.shape[0],))
        else:
            image_size = _sample_image_size(sample)
            if self.zero_images:
                projection = project_vertices(
                    vertices=query_positions,
                    intrinsics=sample["intrinsics"],
                    extrinsics=sample["extrinsics"],
                    image_size=image_size,
                    visibility=sample.get("visibility"),
                )
                valid = projection.valid
                aggregated = query_positions.new_zeros(
                    (query_positions.shape[0], self.projected_image_feature_dim)
                )
                valid_ratio = valid.to(query_positions.dtype).sum(dim=0) / float(
                    max(valid.shape[0], 1)
                )
            else:
                images = sample["images"]
                per_view, valid = self._sample_image_features(
                    images,
                    query_positions,
                    sample["intrinsics"],
                    sample["extrinsics"],
                    image_size,
                    sample.get("visibility"),
                )
                aggregated, valid_ratio = masked_mean_aggregate(per_view, valid)

        edge_index = sample.get("edge_index")
        if edge_index is None:
            edge_index = faces_to_edge_index(sample["faces"], sample["vertices"].shape[0])
        degree = sample.get("vertex_degree")
        if degree is None:
            degree = sample["vertices"].new_zeros((sample["vertices"].shape[0], 1))
            if edge_index.numel() > 0:
                degree.index_add_(
                    0,
                    edge_index[1],
                    torch.ones((edge_index.shape[1], 1), dtype=degree.dtype, device=degree.device),
                )
        if self.geometry_mode == "legacy":
            geometry = torch.cat(
                (
                    query_positions,
                    sample["vertex_normals"],
                    sample["initial_laplacian"],
                    torch.log1p(degree),
                ),
                dim=-1,
            )
        else:
            normalized_positions, position_scale = _normalize_query_positions(
                query_positions, sample
            )
            local_edge_length = sample.get("local_edge_length")
            if local_edge_length is None:
                local_edge_length = mean_incident_edge_length(
                    sample["vertices"].float(), edge_index
                )
            relative_h = local_edge_length.float() / position_scale
            geometry = torch.cat(
                (
                    self.position_encoder(normalized_positions),
                    sample["vertex_normals"].float(),
                    torch.log(relative_h.clamp_min(1e-8)).unsqueeze(-1),
                    torch.log1p(degree.float()),
                ),
                dim=-1,
            )
        ratio_feature = valid_ratio.unsqueeze(-1)
        if self.input_mode == "multiview_only":
            geometry = torch.zeros_like(geometry)
        if self.input_mode == "coarse_only":
            ratio_feature = torch.zeros_like(ratio_feature)
        vertex_features = torch.cat((geometry, ratio_feature, aggregated), dim=-1)
        shared_features, predicted = self.predictor.forward_with_shared_features(
            vertex_features, edge_index
        )
        direct_vertex_displacement_prediction = (
            None
            if self.hybrid_direct_head is None
            else self.hybrid_direct_head(shared_features)
        )
        base_laplacian_prediction = predicted
        oracle_residual_prediction = None
        if self.oracle_residual_expert is not None:
            mask = sample.get("oracle_high_signal_mask")
            if not isinstance(mask, torch.Tensor):
                raise ValueError(
                    "oracle residual expert requires sample.oracle_high_signal_mask."
                )
            if tuple(mask.shape) != (query_positions.shape[0],):
                raise ValueError("oracle_high_signal_mask must have shape [N].")
            oracle_residual_prediction = self.oracle_residual_expert(shared_features)
            oracle_residual_prediction = oracle_residual_prediction * mask.to(
                device=predicted.device, dtype=predicted.dtype
            ).unsqueeze(-1)
            predicted = predicted + oracle_residual_prediction
        dynamic_expert_residual_prediction = None
        dynamic_gate_logit = None
        dynamic_gate_signed = None
        dynamic_gate_effective = None
        if self.dynamic_residual_expert is not None:
            dynamic_expert_residual_prediction = self.dynamic_residual_expert(
                shared_features
            )
            dynamic_gate_logit = self.dynamic_gate_head(shared_features).squeeze(-1)
            dynamic_gate_signed = torch.tanh(dynamic_gate_logit)
            dynamic_gate_effective = torch.relu(dynamic_gate_signed)
            predicted = predicted + dynamic_gate_effective.unsqueeze(
                -1
            ) * dynamic_expert_residual_prediction
        confidence_prediction = (
            None
            if self.confidence_head is None
            else torch.sigmoid(self.confidence_head(vertex_features)).squeeze(-1)
        )
        recovery_lambda_logit = None
        recovery_lambda = None
        if self.recovery_lambda_head is not None:
            recovery_lambda_logit = self.recovery_lambda_head(
                shared_features.mean(dim=0, keepdim=True)
            ).reshape(())
            low = torch.log10(
                recovery_lambda_logit.new_tensor(self.recovery_lambda_minimum)
            )
            high = torch.log10(
                recovery_lambda_logit.new_tensor(self.recovery_lambda_maximum)
            )
            log_lambda = low + torch.sigmoid(recovery_lambda_logit) * (high - low)
            recovery_lambda = torch.pow(recovery_lambda_logit.new_tensor(10.0), log_lambda)
        return LearnedLaplacianOutput(
            predicted_laplacian=predicted,
            confidence_prediction=confidence_prediction,
            vertex_features=vertex_features,
            aggregated_image_features=aggregated,
            valid_view_ratio=valid_ratio,
            valid_views=valid,
            oracle_residual_prediction=oracle_residual_prediction,
            base_laplacian_prediction=base_laplacian_prediction,
            dynamic_expert_residual_prediction=dynamic_expert_residual_prediction,
            dynamic_gate_logit=dynamic_gate_logit,
            dynamic_gate_signed=dynamic_gate_signed,
            dynamic_gate_effective=dynamic_gate_effective,
            recovery_lambda_logit=recovery_lambda_logit,
            recovery_lambda=recovery_lambda,
            direct_vertex_displacement_prediction=direct_vertex_displacement_prediction,
        )

    def architecture_config(self) -> dict[str, Any]:
        first_linear = self.predictor.input_mlp[0]
        return {
            "image_feature_dim": self.image_feature_dim,
            "image_feature_construction_mode": self.image_feature_constructor.mode,
            "image_gaussian_kernel_size": self.image_feature_constructor.gaussian_kernel_size,
            "image_gaussian_sigma": self.image_feature_constructor.gaussian_sigma,
            "image_view_chunk_size": self.image_view_chunk_size,
            "image_gradient_checkpointing": self.image_gradient_checkpointing,
            "image_second_stride": self.image_second_stride,
            "hidden_dim": first_linear.out_features,
            "num_graph_layers": len(self.predictor.blocks),
            "dropout": float(self.predictor.blocks[0].update[2].p),
            "input_mode": self.input_mode,
            "zero_images": self.zero_images,
            "geometry_mode": self.geometry_mode,
            "position_num_frequencies": self.position_encoder.num_frequencies,
            "position_include_input": self.position_encoder.include_input,
            "predict_confidence": self.predict_confidence,
            "oracle_residual_expert_enabled": self.oracle_residual_expert_enabled,
            "oracle_residual_expert_hidden_dim": self.oracle_residual_expert_hidden_dim,
            "dynamic_residual_expert_enabled": self.dynamic_residual_expert_enabled,
            "dynamic_residual_expert_hidden_dim": self.dynamic_residual_expert_hidden_dim,
            "dynamic_gate_hidden_dim": self.dynamic_gate_hidden_dim,
            "dynamic_gate_initial_bias": self.dynamic_gate_initial_bias,
            "recovery_lambda_head_enabled": self.recovery_lambda_head_enabled,
            "recovery_lambda_head_hidden_dim": self.recovery_lambda_head_hidden_dim,
            "recovery_lambda_minimum": self.recovery_lambda_minimum,
            "recovery_lambda_maximum": self.recovery_lambda_maximum,
            "recovery_lambda_initial": self.recovery_lambda_initial,
            "hybrid_direct_head_enabled": self.hybrid_direct_head_enabled,
        }

    def _sample_image_features(
        self,
        images: torch.Tensor,
        query_positions: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: tuple[int, int],
        visibility: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_count = int(images.shape[0])
        chunk_size = self.image_view_chunk_size
        if chunk_size is None or chunk_size >= view_count:
            feature_maps = self.image_feature_constructor(self.image_encoder(images))
            per_view, valid, _ = sample_vertex_features(
                feature_maps=feature_maps,
                vertices=query_positions,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_size=image_size,
                visibility=visibility,
            )
            return per_view, valid

        sampled_chunks = []
        valid_chunks = []
        for start in range(0, view_count, chunk_size):
            stop = min(start + chunk_size, view_count)
            image_chunk = images[start:stop]
            intrinsics_chunk = intrinsics[start:stop]
            extrinsics_chunk = extrinsics[start:stop]
            visibility_chunk = (
                None if visibility is None else visibility[start:stop]
            )

            def make_encoder(
                fixed_visibility: torch.Tensor | None,
            ) -> Any:
                def encode_and_sample(
                    chunk_images: torch.Tensor,
                    chunk_intrinsics: torch.Tensor,
                    chunk_extrinsics: torch.Tensor,
                ) -> torch.Tensor:
                    feature_maps = self.image_feature_constructor(
                        self.image_encoder(chunk_images)
                    )
                    sampled, _, _ = sample_vertex_features(
                        feature_maps=feature_maps,
                        vertices=query_positions,
                        intrinsics=chunk_intrinsics,
                        extrinsics=chunk_extrinsics,
                        image_size=image_size,
                        visibility=fixed_visibility,
                    )
                    return sampled

                return encode_and_sample

            encode_and_sample = make_encoder(visibility_chunk)

            if (
                self.image_gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                sampled = checkpoint(
                    encode_and_sample,
                    image_chunk,
                    intrinsics_chunk,
                    extrinsics_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                sampled = encode_and_sample(
                    image_chunk, intrinsics_chunk, extrinsics_chunk
                )
            projection = project_vertices(
                vertices=query_positions,
                intrinsics=intrinsics_chunk,
                extrinsics=extrinsics_chunk,
                image_size=image_size,
                visibility=visibility_chunk,
            )
            sampled_chunks.append(sampled)
            valid_chunks.append(projection.valid)
        return torch.cat(sampled_chunks, dim=0), torch.cat(valid_chunks, dim=0)


def _normalize_query_positions(
    query_positions: torch.Tensor, sample: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    center = sample.get("position_normalization_center")
    scale = sample.get("position_normalization_scale")
    if center is None or scale is None:
        reference = sample["vertices"].float()
        center = 0.5 * (reference.amin(dim=0) + reference.amax(dim=0))
        scale = torch.linalg.vector_norm(reference - center, dim=-1).amax()
    center = torch.as_tensor(center, dtype=torch.float32, device=query_positions.device)
    scale = torch.as_tensor(
        scale, dtype=torch.float32, device=query_positions.device
    ).reshape(())
    scale = scale.clamp_min(1e-8)
    return (query_positions.float() - center) / scale, scale


def _sample_image_size(sample: Mapping[str, Any]) -> tuple[int, int]:
    images = sample.get("images")
    if isinstance(images, torch.Tensor):
        return int(images.shape[-2]), int(images.shape[-1])
    height = int(sample.get("image_height", 0))
    width = int(sample.get("image_width", 0))
    if height < 1 or width < 1:
        raise ValueError("Samples without image tensors require image_height and image_width.")
    return height, width
