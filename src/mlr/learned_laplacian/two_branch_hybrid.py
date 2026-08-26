from __future__ import annotations

"""Two complete pretrained specialists for continuous hybrid fine-tuning."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .model import LearnedLaplacianModel, LearnedLaplacianOutput


class TwoBranchPretrainedHybridModel(LearnedLaplacianModel):
    """Keep the complete Arm-B and Arm-E networks separate.

    The class deliberately subclasses :class:`LearnedLaplacianModel` so the
    established ragged-mesh trainer and checkpoint format can be reused.  It
    bypasses the parent's constructor: no shared backbone or additional head
    is instantiated.  Arm B supplies the Laplacian latent and Arm E's canonical
    three-vector output is reinterpreted as its already-trained displacement.
    """

    def __init__(
        self,
        arm_b: LearnedLaplacianModel,
        arm_e: LearnedLaplacianModel,
        *,
        arm_b_checkpoint: str | Path,
        arm_e_checkpoint: str | Path,
    ) -> None:
        nn.Module.__init__(self)
        if arm_b.hybrid_direct_head_enabled or arm_e.hybrid_direct_head_enabled:
            raise ValueError("Pretrained B/E specialists must not contain a joint direct head.")
        if arm_b.recovery_lambda_head_enabled or arm_e.recovery_lambda_head_enabled:
            raise ValueError("Continuous B/E hybrid forbids adaptive lambda heads.")
        if arm_b.predict_confidence or arm_e.predict_confidence:
            raise ValueError("Continuous B/E hybrid requires confidence-disabled specialists.")
        for name in (
            "input_mode",
            "zero_images",
            "geometry_mode",
            "image_feature_dim",
            "projected_image_feature_dim",
        ):
            if getattr(arm_b, name) != getattr(arm_e, name):
                raise ValueError(f"Arm B/E architecture mismatch for {name}.")
        self.arm_b = arm_b
        self.arm_e = arm_e
        self.arm_b_checkpoint = str(Path(arm_b_checkpoint))
        self.arm_e_checkpoint = str(Path(arm_e_checkpoint))

        # Trainer contract fields.  The hybrid flag means the second latent is
        # present; it does not imply a shared-backbone hybrid head exists.
        self.input_mode = arm_b.input_mode
        self.zero_images = arm_b.zero_images
        self.geometry_mode = arm_b.geometry_mode
        self.image_feature_dim = arm_b.image_feature_dim
        self.projected_image_feature_dim = arm_b.projected_image_feature_dim
        self.predict_confidence = False
        self.oracle_residual_expert_enabled = False
        self.dynamic_residual_expert_enabled = False
        self.recovery_lambda_head_enabled = False
        self.hybrid_direct_head_enabled = True

    @property
    def image_encoder(self) -> nn.Module:
        """Compatibility alias; both encoders remain separately registered."""

        return self.arm_b.image_encoder

    @property
    def predictor(self) -> nn.Module:
        """The canonical prediction head is the Arm-B Laplacian predictor."""

        return self.arm_b.predictor

    @property
    def hybrid_direct_head(self) -> nn.Module:
        """Expose only Arm E's output MLP for established head diagnostics."""

        return self.arm_e.predictor.output_mlp

    @property
    def recovery_lambda_head(self) -> None:
        return None

    def forward(self, sample: Mapping[str, Any]) -> LearnedLaplacianOutput:
        b_output = self.arm_b(sample)
        e_output = self.arm_e(sample)
        if b_output.direct_vertex_displacement_prediction is not None:
            raise RuntimeError("Arm B unexpectedly emitted a direct latent.")
        if e_output.direct_vertex_displacement_prediction is not None:
            raise RuntimeError("Arm E unexpectedly emitted a joint direct latent.")
        return replace(
            b_output,
            direct_vertex_displacement_prediction=e_output.predicted_laplacian,
        )

    def architecture_config(self) -> dict[str, Any]:
        return {
            "type": "two_complete_pretrained_specialists",
            "shared_backbone": False,
            "arm_b": self.arm_b.architecture_config(),
            "arm_e": self.arm_e.architecture_config(),
            "arm_b_checkpoint": self.arm_b_checkpoint,
            "arm_e_checkpoint": self.arm_e_checkpoint,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def branch_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "b_head": tuple(self.arm_b.predictor.output_mlp.parameters()),
            "b_backbone": tuple(
                parameter
                for name, parameter in self.arm_b.named_parameters()
                if not name.startswith("predictor.output_mlp.")
            ),
            "e_head": tuple(self.arm_e.predictor.output_mlp.parameters()),
            "e_backbone": tuple(
                parameter
                for name, parameter in self.arm_e.named_parameters()
                if not name.startswith("predictor.output_mlp.")
            ),
        }


def load_specialist_checkpoint(
    model: LearnedLaplacianModel,
    checkpoint: str | Path,
) -> None:
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} has no model_state_dict.")
    model.load_state_dict(state, strict=True)
