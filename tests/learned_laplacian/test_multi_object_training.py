import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mlr.laplacian import compute_laplacian_coordinates
from mlr.learned_laplacian import dataset as dataset_module
from mlr.learned_laplacian.dataset import save_prepared_sample
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian import multi_dataset as multi_dataset_module
from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian import multi_trainer
from mlr.learned_laplacian.multi_trainer import (
    _EpochIndexSampler,
    _MaterializedPreparedDataset,
    _build_lr_scheduler,
    _prediction_loss_inputs,
    _prediction_loss_space,
    _prepare_object_static,
    _select_prepared_views,
    train_multi_object,
)
from mlr.learned_laplacian.target_scaling import normalize_laplacian_by_edge_scale

from .helpers import tiny_sample


def _triangle_sample(sample_id: str) -> dict:
    sample = tiny_sample()
    vertices = sample["vertices"][:3].clone()
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    target_positions = vertices.clone()
    target_positions[:, 2] += torch.tensor([0.03, -0.02, 0.01])
    sample.update(
        {
            "sample_id": sample_id,
            "vertices": vertices,
            "faces": faces,
            "vertex_normals": torch.nn.functional.normalize(
                vertices - vertices.mean(dim=0), dim=-1
            ),
            "initial_laplacian": torch.from_numpy(
                compute_laplacian_coordinates(vertices.numpy(), faces.numpy(), "uniform")
            ).float(),
            "laplacian_target": torch.from_numpy(
                compute_laplacian_coordinates(
                    target_positions.numpy(), faces.numpy(), "uniform"
                )
            ).float(),
            "target_confidence": torch.ones(3),
            "visibility": torch.ones((1, 3), dtype=torch.bool),
            "target_positions": target_positions,
            "gt_vertices": target_positions,
            "gt_faces": faces,
        }
    )
    return sample


def _multi_config() -> dict:
    return {
        "seed": 11,
        "device": "cpu",
        "input_mode": "coarse_only",
        "target_mode": "edge_scale_normalized_laplacian",
        "target_scaling": {
            "method": "square_of_mean_incident_edge_length",
            "epsilon": 1e-12,
            "clip_max_norm": None,
        },
        "image_encoder": {"feature_dim": 8},
        "model": {"hidden_dim": 16, "num_graph_layers": 1, "dropout": 0.0},
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "loss": "huber",
            "huber_delta": 0.1,
            "gradient_clip_norm": 1.0,
        },
        "multi_object_training": {
            "epochs": 4,
            "gradient_accumulation_meshes": 2,
            "shuffle": True,
            "validation_every_epochs": 2,
            "checkpoint_every_epochs": 0,
        },
    }


def test_prediction_loss_space_defaults_to_output_representation() -> None:
    assert _prediction_loss_space({}) == "output_representation"
    with pytest.raises(ValueError, match="prediction_loss_space"):
        _prediction_loss_space({"prediction_loss_space": "unknown"})


def test_normalized_output_can_be_compared_in_raw_loss_space() -> None:
    h = torch.tensor([0.5, 2.0])
    prediction_hat = torch.tensor([[2.0, -1.0, 0.5], [3.0, 4.0, -2.0]], requires_grad=True)
    target_raw = torch.tensor([[0.4, -0.3, 0.2], [8.0, 12.0, -4.0]])
    prepared = multi_trainer._PreparedObject(
        sample={"local_edge_length": h},
        training_target=target_raw / h.square().unsqueeze(-1),
        clipped_target_vertices=0,
        raw_target=target_raw,
    )

    prediction, target = _prediction_loss_inputs(
        prediction_hat,
        prepared,
        target_mode="edge_scale_normalized_laplacian",
        epsilon=1e-12,
        prediction_loss_space="raw_laplacian",
    )

    assert torch.allclose(prediction, prediction_hat * h.square().unsqueeze(-1))
    assert torch.equal(target, target_raw)
    prediction.sum().backward()
    assert torch.allclose(
        prediction_hat.grad, h.square().unsqueeze(-1).expand_as(prediction_hat)
    )


def test_raw_output_is_not_rescaled_for_raw_loss_space() -> None:
    prediction_raw = torch.randn(4, 3)
    target_raw = torch.randn(4, 3)
    prepared = multi_trainer._PreparedObject(
        sample={"local_edge_length": torch.ones(4)},
        training_target=target_raw,
        clipped_target_vertices=0,
        raw_target=target_raw,
    )

    prediction, target = _prediction_loss_inputs(
        prediction_raw,
        prepared,
        target_mode="raw_laplacian",
        epsilon=1e-12,
        prediction_loss_space="raw_laplacian",
    )

    assert prediction is prediction_raw
    assert torch.equal(target, target_raw)


def test_normalized_output_raw_loss_trains_through_prepared_loader() -> None:
    config = _multi_config()
    config["training"]["prediction_loss_space"] = "raw_laplacian"
    config["multi_object_training"].update(
        {"epochs": 1, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("raw_loss_train")],
        [_triangle_sample("raw_loss_validation")],
        config,
        progress=False,
    )

    assert result.prediction_loss_space == "raw_laplacian"
    assert math.isfinite(result.final_train_loss)
    assert result.final_validation_loss is not None
    assert math.isfinite(result.final_validation_loss)


def _recovery_aware_sample(sample_id: str) -> dict:
    sample = _triangle_sample(sample_id)
    sample["clean_reference_vertices"] = sample["target_positions"].clone()
    sample["clean_reference_faces"] = sample["faces"].clone()
    return sample


def _recovery_aware_config() -> dict:
    config = _multi_config()
    config["target_mode"] = "raw_laplacian"
    config["confidence"] = {"enabled": False}
    config["training"]["vertex_sampling"] = {"mode": "full"}
    config["training"]["recovery_aware_geometry_loss"] = {
        "enabled": True,
        "lambda": 1e-2,
        "beta": 0.1,
        "maximum_iterations": 128,
        "tolerance": 1e-4,
        "runtime_diagnostics": True,
    }
    config["multi_object_training"].update(
        {
            "epochs": 1,
            "gradient_accumulation_meshes": 1,
            "validation_every_epochs": 1,
        }
    )
    return config


def test_recovery_aware_clean_vertices_are_loss_side_only() -> None:
    prepared = _prepare_object_static(
        _recovery_aware_sample("loss_side_clean"), _recovery_aware_config()
    )
    assert prepared.clean_vertices is not None
    assert "clean_reference_vertices" not in prepared.sample
    assert "clean_reference_faces" not in prepared.sample
    assert not any("clean" in key or key.startswith("gt_") for key in prepared.sample)


def test_recovery_aware_training_reports_geometry_loss(tmp_path) -> None:
    config = _recovery_aware_config()
    config["multi_object_training"]["report_every_optimizer_steps"] = 1
    result = train_multi_object(
        [_recovery_aware_sample("recovery_train")],
        [_recovery_aware_sample("recovery_validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )
    record = result.history[-1]
    assert record["train_recovery_refine_loss"] is not None
    assert record["validation_recovery_refine_loss"] is not None
    assert record["validation_recovered_vertex_rms"] is not None
    assert math.isfinite(float(record["train_recovery_refine_loss"]))
    assert math.isfinite(float(record["validation_recovery_refine_loss"]))
    assert math.isfinite(float(record["validation_recovered_vertex_rms"]))
    step_record = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )[-1]
    assert float(step_record["pcg_iterations_mean"]) > 0
    assert float(step_record["pcg_iterations_max"]) >= float(
        step_record["pcg_iterations_mean"]
    )
    assert float(step_record["pcg_relative_residual_max"]) <= 1.05e-4
    assert int(step_record["pcg_failed_solves"]) == 0
    assert float(step_record["delta_pred_gradient_norm"]) > 0
    assert float(step_record["prediction_head_gradient_norm"]) > 0
    assert int(step_record["nan_inf_count"]) == 0


def test_recovery_only_objective_backpropagates_through_sparse_solve(tmp_path) -> None:
    config = _recovery_aware_config()
    config["training"]["recovery_aware_geometry_loss"].update(
        {
            "prediction_loss_weight": 0.0,
            "beta": 1.0,
        }
    )
    config["multi_object_training"]["report_every_optimizer_steps"] = 1
    result = train_multi_object(
        [_recovery_aware_sample("recovery_only_train")],
        [_recovery_aware_sample("recovery_only_validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )
    record = result.history[-1]
    assert math.isclose(
        float(record["train_objective"]),
        float(record["train_recovery_refine_loss"]),
        rel_tol=1e-6,
        abs_tol=1e-10,
    )
    assert math.isclose(
        float(record["validation_loss"]),
        float(record["validation_recovery_refine_loss"]),
        rel_tol=1e-6,
        abs_tol=1e-10,
    )
    step = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )[-1]
    assert math.isclose(
        float(step["train_objective"]),
        float(step["train_recovery_refine_loss"]),
        rel_tol=1e-6,
        abs_tol=1e-10,
    )
    assert float(step["delta_pred_gradient_norm"]) > 0
    assert float(step["prediction_head_gradient_norm"]) > 0
    assert int(step["pcg_failed_solves"]) == 0
    assert int(step["nan_inf_count"]) == 0


def test_hybrid_single_loss_trains_both_heads_and_selects_validation_chamfer(
    tmp_path,
) -> None:
    config = _multi_config()
    config["target_mode"] = "raw_laplacian"
    config["confidence"] = {"enabled": False}
    config["model"]["hybrid_direct_head"] = {"enabled": True}
    config["model"]["recovery_lambda_head"] = {"enabled": False}
    config["training"]["vertex_sampling"] = {"mode": "full"}
    config["training"]["recovery_aware_geometry_loss"] = {"enabled": False}
    config["training"]["hybrid_single_geometry_loss"] = {
        "enabled": True,
        "lambda": 3e-2,
        "maximum_iterations": 256,
        "tolerance": 1e-8,
        "compute_dtype": "float64",
        "runtime_diagnostics": True,
        "validation_surface_samples": 50,
    }
    config["multi_object_training"].update(
        {
            "epochs": 1,
            "gradient_accumulation_meshes": 1,
            "validation_every_epochs": 1,
            "report_every_optimizer_steps": 1,
        }
    )
    result = train_multi_object(
        [_recovery_aware_sample("hybrid_train")],
        [_recovery_aware_sample("hybrid_validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )
    record = result.history[-1]
    assert record["validation_hybrid_chamfer"] is not None
    assert result.best_selection_loss == pytest.approx(
        float(record["validation_hybrid_chamfer"])
    )
    step = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )[-1]
    assert float(step["delta_pred_gradient_norm"]) > 0
    assert float(step["v_direct_gradient_norm"]) > 0
    assert float(step["prediction_head_gradient_norm"]) > 0
    assert float(step["direct_head_gradient_norm"]) > 0
    assert float(step["laplacian_output_rms"]) > 0
    assert int(step["pcg_failed_solves"]) == 0
    assert int(step["nan_inf_count"]) == 0


def test_two_complete_pretrained_branches_remain_separate_and_receive_gradients(
    tmp_path,
) -> None:
    config = _multi_config()
    config["target_mode"] = "raw_laplacian"
    config["confidence"] = {"enabled": False}
    config["model"]["hybrid_direct_head"] = {"enabled": False}
    config["model"]["recovery_lambda_head"] = {"enabled": False}
    base = multi_trainer._build_single_model(config, None, False)
    single_parameters = sum(parameter.numel() for parameter in base.parameters())
    b_checkpoint = tmp_path / "arm_b.pt"
    e_checkpoint = tmp_path / "arm_e.pt"
    torch.save({"model_state_dict": base.state_dict()}, b_checkpoint)
    e_state = {
        name: value.clone() for name, value in base.state_dict().items()
    }
    e_state["predictor.output_mlp.2.bias"] += 0.01
    torch.save({"model_state_dict": e_state}, e_checkpoint)

    config["model"]["two_branch_pretrained_hybrid"] = {
        "enabled": True,
        "arm_b_checkpoint": str(b_checkpoint),
        "arm_e_checkpoint": str(e_checkpoint),
    }
    config["training"]["vertex_sampling"] = {"mode": "full"}
    config["training"]["recovery_aware_geometry_loss"] = {"enabled": False}
    config["training"]["hybrid_single_geometry_loss"] = {
        "enabled": True,
        "lambda": 3e-2,
        "maximum_iterations": 256,
        "tolerance": 1e-8,
        "compute_dtype": "float64",
        "runtime_diagnostics": True,
        "validation_surface_samples": 50,
    }
    config["multi_object_training"].update(
        {
            "epochs": 1,
            "gradient_accumulation_meshes": 1,
            "validation_every_epochs": 1,
            "report_every_optimizer_steps": 1,
        }
    )
    model = multi_trainer._build_model(config, None, False)
    assert model.arm_b is not model.arm_e
    assert sum(parameter.numel() for parameter in model.parameters()) == 2 * single_parameters
    assert torch.equal(
        model.arm_b.predictor.output_mlp[2].bias,
        base.predictor.output_mlp[2].bias,
    )
    assert torch.equal(
        model.arm_e.predictor.output_mlp[2].bias,
        base.predictor.output_mlp[2].bias + 0.01,
    )

    result = train_multi_object(
        [_recovery_aware_sample("two_branch_train")],
        [_recovery_aware_sample("two_branch_validation")],
        config,
        output_dir=tmp_path / "run",
        progress=False,
    )
    assert sum(parameter.numel() for parameter in result.model.parameters()) == 2 * single_parameters
    step = json.loads(
        (tmp_path / "run" / "training_step_history.json").read_text(
            encoding="utf-8"
        )
    )[-1]
    for key in (
        "delta_pred_gradient_norm",
        "v_direct_gradient_norm",
        "b_laplacian_head_gradient_norm",
        "b_backbone_gradient_norm",
        "e_direct_head_gradient_norm",
        "e_backbone_gradient_norm",
    ):
        assert float(step[key]) > 0
    assert int(step["nan_inf_count"]) == 0


def test_cotangent_hybrid_single_loss_preserves_dual_gradient_contract(
    tmp_path,
) -> None:
    config = _multi_config()
    config["target_mode"] = "raw_laplacian"
    config["confidence"] = {"enabled": False}
    config["model"]["hybrid_direct_head"] = {"enabled": True}
    config["model"]["recovery_lambda_head"] = {"enabled": False}
    config["training"]["vertex_sampling"] = {"mode": "full"}
    config["training"]["recovery_aware_geometry_loss"] = {"enabled": False}
    config["training"]["hybrid_single_geometry_loss"] = {
        "enabled": True,
        "operator": "symmetric_cotangent_stiffness",
        "cotangent_relative_area_epsilon": 1e-12,
        "lambda": 3e-2,
        "maximum_iterations": 256,
        "tolerance": 1e-8,
        "compute_dtype": "float64",
        "runtime_diagnostics": True,
        "validation_surface_samples": 50,
    }
    config["multi_object_training"].update(
        {
            "epochs": 1,
            "gradient_accumulation_meshes": 1,
            "validation_every_epochs": 1,
            "report_every_optimizer_steps": 1,
        }
    )
    static = _prepare_object_static(
        _recovery_aware_sample("cotangent_static"), config
    )
    assert "cotangent_edge_index" in static.sample
    assert "cotangent_edge_weight" in static.sample
    assert "cotangent_diagonal" in static.sample
    result = train_multi_object(
        [_recovery_aware_sample("cotangent_train")],
        [_recovery_aware_sample("cotangent_validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )
    assert math.isfinite(result.final_train_loss)
    step = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )[-1]
    assert float(step["delta_pred_gradient_norm"]) > 0
    assert float(step["v_direct_gradient_norm"]) > 0
    assert float(step["prediction_head_gradient_norm"]) > 0
    assert float(step["direct_head_gradient_norm"]) > 0
    assert int(step["pcg_failed_solves"]) == 0
    assert int(step["nan_inf_count"]) == 0


def test_hard_anchor_lambda0_training_uses_graph_only_anchors(tmp_path) -> None:
    config = _recovery_aware_config()
    config["training"]["recovery_aware_geometry_loss"].update(
        {
            "solver": "hard_anchor_lambda0",
            "lambda": 0.0,
            "compute_dtype": "float64",
            "maximum_iterations": 256,
            "tolerance": 1e-7,
        }
    )
    config["multi_object_training"]["report_every_optimizer_steps"] = 1
    prepared = _prepare_object_static(
        _recovery_aware_sample("hard_anchor_static"), config
    )
    assert prepared.sample["hard_anchor_indices"].tolist() == [0]
    assert "clean_reference_vertices" not in prepared.sample
    result = train_multi_object(
        [_recovery_aware_sample("hard_anchor_train")],
        [_recovery_aware_sample("hard_anchor_validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )
    assert math.isfinite(result.final_train_loss)
    step = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )[-1]
    assert int(step["pcg_failed_solves"]) == 0
    assert float(step["pcg_relative_residual_max"]) <= 1.05e-7
    assert float(step["delta_pred_gradient_norm"]) > 0
    assert float(step["prediction_head_gradient_norm"]) > 0
    assert int(step["nan_inf_count"]) == 0


def test_adaptive_recovery_lambda_training_reports_finite_nonzero_gradients(
    tmp_path,
) -> None:
    config = _recovery_aware_config()
    config["model"]["recovery_lambda_head"] = {
        "enabled": True,
        "hidden_dim": 4,
        "lambda_min": 1e-3,
        "lambda_max": 1e-1,
        "lambda_initial": 1e-2,
    }
    config["training"]["recovery_aware_geometry_loss"]["adaptive_lambda"] = True
    config["multi_object_training"]["report_every_optimizer_steps"] = 1
    train_multi_object(
        [_recovery_aware_sample("adaptive_lambda_train")],
        [_recovery_aware_sample("adaptive_lambda_validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )
    step = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )[-1]
    assert 1e-3 < float(step["recovery_lambda_mean"]) < 1e-1
    assert float(step["recovery_lambda_gradient_norm"]) > 0
    assert float(step["recovery_lambda_head_gradient_norm"]) > 0
    assert int(step["nan_inf_count"]) == 0


def _multi_view_sample(num_views: int = 14) -> dict:
    sample = tiny_sample()
    sample["sample_id"] = "multi_view"
    sample["images"] = torch.stack(
        [torch.full((3, 16, 16), index / 20.0) for index in range(num_views)]
    )
    sample["image_paths"] = [f"view_{index:02d}.png" for index in range(num_views)]
    sample["intrinsics"] = sample["intrinsics"].repeat(num_views, 1, 1)
    sample["extrinsics"] = sample["extrinsics"].repeat(num_views, 1, 1)
    sample["visibility"] = sample["visibility"].repeat(num_views, 1)
    for index in range(num_views):
        sample["intrinsics"][index, 0, 1] = float(index)
        sample["extrinsics"][index, 0, 3] = float(index)
        sample["visibility"][index] = torch.tensor(
            [(index >> bit) & 1 for bit in range(4)], dtype=torch.bool
        )
    return sample


def test_manifest_lazily_loads_variable_topology_splits(tmp_path):
    train = tiny_sample()
    train["sample_id"] = "tetra_train"
    validation = _triangle_sample("triangle_validation")
    save_prepared_sample(train, tmp_path / "train.pt")
    save_prepared_sample(validation, tmp_path / "validation.pt")
    manifest = {
        "samples": [
            {"sample_id": "tetra_train", "path": "train.pt", "split": "train"},
            {
                "sample_id": "triangle_validation",
                "path": "validation.pt",
                "split": "validation",
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")

    assert train_dataset.sample_ids == ("tetra_train",)
    assert validation_dataset.sample_ids == ("triangle_validation",)
    assert train_dataset[0]["vertices"].shape[0] == 4
    assert validation_dataset[0]["vertices"].shape[0] == 3
    validate_disjoint_splits(train_dataset, validation_dataset)


def test_manifest_dataset_root_preserves_lazy_image_resolution(tmp_path):
    dataset_root = tmp_path / "source_dataset"
    dataset_root.mkdir()
    train = tiny_sample()
    train["sample_id"] = "portable_train"
    save_prepared_sample(train, dataset_root / "train.pt")
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    manifest_path = relocated / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "samples": [
                    {
                        "sample_id": "portable_train",
                        "path": str(dataset_root / "train.pt"),
                        "split": "train",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    assert dataset.records[0].dataset_root == dataset_root.resolve()


def test_manifest_dataset_loads_each_file_only_once(tmp_path, monkeypatch):
    train = tiny_sample()
    train["sample_id"] = "train"
    validation = _triangle_sample("validation")
    save_prepared_sample(train, tmp_path / "train.pt")
    save_prepared_sample(validation, tmp_path / "validation.pt")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"path": "train.pt", "split": "train"},
                    {"path": "validation.pt", "split": "validation"},
                ]
            }
        ),
        encoding="utf-8",
    )
    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")
    original_load = multi_dataset_module.load_prepared_sample
    loaded_paths = []

    def counted_load(path, **kwargs):
        loaded_paths.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(multi_dataset_module, "load_prepared_sample", counted_load)
    validate_disjoint_splits(train_dataset, validation_dataset)
    train_sample = train_dataset[0]
    validation_sample = validation_dataset[0]

    assert len(loaded_paths) == 2
    assert train_sample is train_dataset[0]
    assert validation_sample is validation_dataset[0]


def test_manifest_rejects_train_validation_path_leakage(tmp_path):
    sample = tiny_sample()
    save_prepared_sample(sample, tmp_path / "shared.pt")
    manifest = {
        "samples": [
            {"path": "shared.pt", "split": "train"},
            {"path": "shared.pt", "split": "validation"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")

    with pytest.raises(ValueError, match="appears in both"):
        validate_disjoint_splits(train_dataset, validation_dataset)


def test_distributed_epoch_sampler_shards_and_pads_deterministically():
    shards = []
    for rank in range(3):
        generator = torch.Generator().manual_seed(29)
        sampler = _EpochIndexSampler(
            7,
            shuffle=True,
            generator=generator,
            rank=rank,
            world_size=3,
        )
        sampler.set_epoch(4)
        epoch_indices = list(sampler)
        shard = [index for index, _ in epoch_indices]
        assert all(epoch == 4 for _, epoch in epoch_indices)
        assert len(shard) == 3
        shards.append(shard)

    flattened = [index for shard in shards for index in shard]
    assert set(flattened) == set(range(7))
    assert len(flattened) == 9


def test_torchrun_two_process_training_writes_canonical_rank_zero_outputs(tmp_path):
    training_samples = [
        _triangle_sample(f"train_{suffix}") for suffix in ("a", "b", "c", "d")
    ]
    validation = _triangle_sample("validation")
    for sample in (*training_samples, validation):
        save_prepared_sample(sample, tmp_path / f"{sample['sample_id']}.pt")
    manifest = {
        "samples": [
            *[
                {
                    "sample_id": sample["sample_id"],
                    "path": f"{sample['sample_id']}.pt",
                    "split": "train",
                }
                for sample in training_samples
            ],
            {
                "sample_id": "validation",
                "path": "validation.pt",
                "split": "validation",
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = _multi_config()
    config["multi_object_training"].update(
        {
            "epochs": 1,
            "gradient_accumulation_meshes": 2,
            "validation_every_epochs": 1,
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "output"
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(repository / "scripts/train_multi_mesh_laplacian.py"),
            "--manifest",
            str(manifest_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("distributed training:") == 1
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["distributed_world_size"] == 2
    assert metrics["global_batch_meshes"] == 4
    assert metrics["optimizer_steps"] == 1
    assert set(metrics["per_object_metrics"]["train"]) == {
        sample["sample_id"] for sample in training_samples
    }
    assert len(list((output_dir / "predictions" / "train").glob("*_raw_delta.npy"))) == 4
    checkpoint = torch.load(output_dir / "checkpoint_latest.pt", weights_only=False)
    assert checkpoint["optimizer_steps"] == 1
    assert all(
        not name.startswith("module.") for name in checkpoint["model_state_dict"]
    )


def test_shared_model_trains_across_different_mesh_topologies(tmp_path):
    tetra = tiny_sample()
    tetra["sample_id"] = "tetra_train"
    triangle = _triangle_sample("triangle_train")
    validation = _triangle_sample("triangle_validation")

    result = train_multi_object(
        [tetra, triangle],
        [validation],
        _multi_config(),
        output_dir=tmp_path,
        progress=False,
    )

    assert result.optimizer_steps == 4
    assert result.target_mode == "edge_scale_normalized_laplacian"
    assert 1 <= result.best_epoch <= 4
    assert math.isfinite(result.best_selection_loss)
    assert math.isfinite(result.final_train_loss)
    assert math.isfinite(result.final_validation_loss)
    assert result.per_object_metrics["train"]["tetra_train"]["vertex_count"] == 4
    assert result.per_object_metrics["train"]["triangle_train"]["vertex_count"] == 3
    assert result.per_object_metrics["validation"]["triangle_validation"]["face_count"] == 1
    for split_metrics in result.per_object_metrics.values():
        for metrics in split_metrics.values():
            assert all(
                math.isfinite(value)
                for value in metrics["recovered_raw_space"].values()
            )
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions" / "train" / "tetra_train_raw_delta.npy").is_file()
    triangle_prediction = np.load(
        tmp_path / "predictions" / "train" / "triangle_train_raw_delta.npy"
    )
    assert triangle_prediction.shape == (3, 3)


def test_static_preparation_runs_once_per_sample_across_epochs(monkeypatch):
    samples = [tiny_sample(), _triangle_sample("triangle_train")]
    samples[0]["sample_id"] = "tetra_train"
    validation = [_triangle_sample("triangle_validation")]
    calls = {"validate": 0, "edges": 0, "normalize": 0}
    original_validate = multi_trainer.validate_sample
    original_edges = dataset_module.faces_to_edge_index
    original_normalize = dataset_module.normalize_laplacian_by_edge_scale

    def counted_validate(sample):
        calls["validate"] += 1
        return original_validate(sample)

    def counted_edges(faces, num_vertices=None):
        calls["edges"] += 1
        return original_edges(faces, num_vertices)

    def counted_normalize(*args, **kwargs):
        calls["normalize"] += 1
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(multi_trainer, "validate_sample", counted_validate)
    monkeypatch.setattr(dataset_module, "faces_to_edge_index", counted_edges)
    monkeypatch.setattr(dataset_module, "normalize_laplacian_by_edge_scale", counted_normalize)

    train_multi_object(samples, validation, _multi_config(), progress=False)

    assert calls == {"validate": 3, "edges": 3, "normalize": 3}


def test_static_prepared_fields_match_reference_calculation():
    config = _multi_config()
    config["target_scaling"]["clip_max_norm"] = 0.05
    sample = _triangle_sample("prepared_triangle")
    prepared = _prepare_object_static(sample, config)
    expected_edges = faces_to_edge_index(sample["faces"], sample["vertices"].shape[0])
    expected_degree = sample["vertices"].new_zeros((3, 1))
    expected_degree.index_add_(
        0,
        expected_edges[1],
        torch.ones((expected_edges.shape[1], 1)),
    )
    expected_target = normalize_laplacian_by_edge_scale(
        prepared.raw_target,
        prepared.sample["local_edge_length"],
        eps=config["target_scaling"]["epsilon"],
        valid_scale_mask=prepared.sample["valid_scale_mask"],
    )
    magnitudes = torch.linalg.vector_norm(expected_target, dim=-1)
    expected_clipped = magnitudes > 0.05
    expected_target = expected_target * (
        0.05 / magnitudes.clamp_min(1e-12)
    ).clamp_max(1.0).unsqueeze(-1)

    assert torch.equal(prepared.sample["edge_index"], expected_edges)
    assert torch.equal(prepared.sample["vertex_degree"], expected_degree)
    assert torch.equal(prepared.sample["valid_scale_mask"], torch.ones(3, dtype=torch.bool))
    assert torch.allclose(prepared.training_target, expected_target)
    assert prepared.clipped_target_vertices == int(expected_clipped.sum())


def test_device_cache_switch_preserves_cpu_training_results():
    def run(cache_on_device: bool):
        config = _multi_config()
        config["multi_object_training"]["cache_prepared_samples_on_device"] = cache_on_device
        return train_multi_object(
            [_triangle_sample("train")],
            [_triangle_sample("validation")],
            config,
            progress=False,
        )

    cached = run(True)
    streamed = run(False)

    assert [item["train_loss"] for item in cached.history] == pytest.approx(
        [item["train_loss"] for item in streamed.history]
    )
    assert [item["validation_loss"] for item in cached.history] == pytest.approx(
        [item["validation_loss"] for item in streamed.history]
    )
    assert cached.best_epoch == streamed.best_epoch
    assert cached.best_selection_loss == pytest.approx(streamed.best_selection_loss)


def test_validation_schedule_and_timing_metrics(tmp_path):
    config = _multi_config()
    config["multi_object_training"].update(
        {"epochs": 5, "validation_every_epochs": 3}
    )
    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )

    validated_epochs = [
        item["epoch"] for item in result.history if item["validation_loss"] is not None
    ]
    assert validated_epochs == [1, 3, 5]
    assert all(item["train_seconds"] >= 0 for item in result.history)
    assert result.static_preparation_seconds >= 0
    assert result.device_cache_seconds >= 0
    assert result.mean_epoch_train_seconds >= 0
    assert result.mean_validation_seconds >= 0
    assert result.mean_epoch_data_loading_seconds >= 0
    assert result.mean_epoch_gpu_transfer_seconds >= 0
    assert result.mean_epoch_forward_backward_seconds >= 0
    assert result.mean_epoch_image_decode_resize_seconds >= 0
    assert result.mean_train_views_per_sample >= 1
    assert result.mean_epoch_decoded_image_bytes >= 0
    assert result.mean_optimizer_step_seconds >= 0
    assert result.peak_cpu_memory_mb > 0
    assert not result.cuda_transfer_overlap_enabled
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mean_epoch_train_seconds"] == result.mean_epoch_train_seconds
    assert metrics["peak_cpu_memory_mb"] == result.peak_cpu_memory_mb
    assert not metrics["cuda_transfer_overlap_enabled"]
    assert all(
        key in result.history[0]
        for key in (
            "data_loading_seconds",
            "sample_wait_seconds",
            "image_decode_resize_seconds",
            "mean_image_decode_resize_seconds_per_mesh",
            "mean_used_view_count",
            "decoded_image_bytes",
            "gpu_transfer_seconds",
            "pin_or_transfer_seconds",
            "forward_backward_seconds",
            "total_step_seconds",
        )
    )
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=False)
    assert set(checkpoint) == {
        "epoch",
        "train_loss",
        "validation_loss",
        "model_config",
        "model_state_dict",
        "optimizer_state_dict",
        "experiment_config",
        "train_meshes",
        "validation_meshes",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("cache_on_device", [True, False])
def test_cuda_device_cache_paths(cache_on_device):
    config = _multi_config()
    config["device"] = "cuda"
    config["multi_object_training"].update(
        {"epochs": 1, "cache_prepared_samples_on_device": cache_on_device}
    )

    result = train_multi_object(
        [_triangle_sample("train")], (), config, progress=False
    )

    assert result.device == "cuda"
    assert math.isfinite(result.final_train_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_keeps_training_loss_finite():
    config = _multi_config()
    config["device"] = "cuda"
    config["input_mode"] = "coarse_plus_multiview"
    config["training"]["amp"] = {"enabled": True, "dtype": "float16"}
    config["multi_object_training"].update(
        {"epochs": 1, "cache_prepared_samples_on_device": False}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert result.device == "cuda"
    assert result.amp_enabled
    assert math.isfinite(result.final_train_loss)
    assert math.isfinite(result.final_validation_loss)


def test_cuda_transfer_overlap_requires_prefetchable_host_data():
    settings = multi_trainer._data_loader_settings(
        {"data_loading": {"pin_memory": True}}
    )

    assert multi_trainer._cuda_transfer_overlap_enabled(
        torch.device("cuda"), cache_on_device=False, settings=settings
    )
    assert not multi_trainer._cuda_transfer_overlap_enabled(
        torch.device("cpu"), cache_on_device=False, settings=settings
    )
    assert not multi_trainer._cuda_transfer_overlap_enabled(
        torch.device("cuda"), cache_on_device=True, settings=settings
    )

    disabled = multi_trainer._data_loader_settings(
        {"data_loading": {"pin_memory": True, "cuda_prefetch": False}}
    )
    assert not multi_trainer._cuda_transfer_overlap_enabled(
        torch.device("cuda"), cache_on_device=False, settings=disabled
    )


def test_data_loader_supports_low_fd_multiprocessing_sharing_strategy(monkeypatch):
    settings = multi_trainer._data_loader_settings(
        {
            "data_loading": {
                "num_workers": 4,
                "multiprocessing_sharing_strategy": "file_system",
            }
        }
    )
    configured = []
    monkeypatch.setattr(
        torch.multiprocessing,
        "get_sharing_strategy",
        lambda: "file_descriptor",
    )
    monkeypatch.setattr(
        torch.multiprocessing,
        "set_sharing_strategy",
        configured.append,
    )

    multi_trainer._configure_multiprocessing_sharing(settings)

    assert settings.multiprocessing_sharing_strategy == "file_system"
    assert configured == ["file_system"]


def test_data_loader_rejects_unknown_multiprocessing_sharing_strategy():
    with pytest.raises(ValueError, match="multiprocessing_sharing_strategy"):
        multi_trainer._data_loader_settings(
            {
                "data_loading": {
                    "multiprocessing_sharing_strategy": "unknown",
                }
            }
        )


def _use_constant_validation(monkeypatch, loss=1.0):
    calls = []

    def constant_validation(*args, **kwargs):
        calls.append(loss)
        return loss, {}

    monkeypatch.setattr(multi_trainer, "_evaluate_dataset", constant_validation)
    return calls


def test_scheduler_disabled_or_omitted_preserves_training(monkeypatch):
    _use_constant_validation(monkeypatch)

    def run(scheduler_config):
        config = _multi_config()
        if scheduler_config is not None:
            config["training"]["lr_scheduler"] = scheduler_config
        return train_multi_object(
            [_triangle_sample("train")],
            [_triangle_sample("validation")],
            config,
            progress=False,
        )

    omitted = run(None)
    disabled = run({"type": "none"})

    assert [item["train_loss"] for item in omitted.history] == pytest.approx(
        [item["train_loss"] for item in disabled.history]
    )
    assert omitted.optimizer_steps == disabled.optimizer_steps
    assert omitted.best_epoch == disabled.best_epoch
    assert omitted.best_selection_loss == disabled.best_selection_loss
    assert {item["learning_rate"] for item in omitted.history} == {0.01}
    assert {item["learning_rate"] for item in disabled.history} == {0.01}
    assert omitted.lr_scheduler_type == disabled.lr_scheduler_type == "none"


def test_reduce_on_plateau_uses_validation_patience(monkeypatch):
    _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"].update(
        {
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience_validations": 1,
                "threshold": 0.0,
                "threshold_mode": "abs",
                "cooldown_validations": 0,
                "min_lr": 1e-6,
            },
        }
    )
    config["multi_object_training"].update(
        {"epochs": 4, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert [item["learning_rate"] for item in result.history] == pytest.approx(
        [0.001, 0.001, 0.0005, 0.0005]
    )
    assert result.lr_reduction_count == 1


def test_scheduler_steps_only_on_validation_epochs(monkeypatch):
    validation_calls = _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"].update(
        {
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience_validations": 0,
                "threshold": 0.0,
                "min_lr": 1e-6,
            },
        }
    )
    config["multi_object_training"].update(
        {"epochs": 6, "validation_every_epochs": 5}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert [item["validation_loss"] is not None for item in result.history] == [
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert [item["learning_rate"] for item in result.history] == pytest.approx(
        [0.001, 0.001, 0.001, 0.001, 0.0005, 0.00025]
    )
    # Three scheduled validations plus the final train and validation evaluations.
    assert len(validation_calls) == 5


def test_scheduler_respects_minimum_learning_rate(monkeypatch):
    _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"].update(
        {
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.1,
                "patience_validations": 0,
                "threshold": 0.0,
                "cooldown_validations": 0,
                "min_lr": 1e-6,
            },
        }
    )
    config["multi_object_training"].update(
        {"epochs": 8, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    learning_rates = [item["learning_rate"] for item in result.history]
    assert min(learning_rates) == pytest.approx(1e-6)
    assert result.final_learning_rate == pytest.approx(1e-6)
    assert all(rate >= 1e-6 for rate in learning_rates)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"factor": 0.0}, "factor"),
        ({"factor": 1.0}, "factor"),
        ({"patience_validations": -1}, "patience_validations"),
        ({"threshold": -1.0}, "threshold"),
        ({"threshold_mode": "invalid"}, "threshold_mode"),
        ({"cooldown_validations": -1}, "cooldown_validations"),
        ({"min_lr": -1.0}, "min_lr"),
        ({"type": "cosine"}, "Unsupported lr_scheduler type"),
    ],
)
def test_scheduler_rejects_invalid_configuration(override, message):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler_config = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience_validations": 1,
        "threshold": 0.0,
        "threshold_mode": "abs",
        "cooldown_validations": 0,
        "min_lr": 1e-6,
    }
    scheduler_config.update(override)

    with pytest.raises(ValueError, match=message):
        _build_lr_scheduler(optimizer, {"lr_scheduler": scheduler_config})


def test_scheduler_history_metrics_and_reduction_log(tmp_path, monkeypatch, capsys):
    _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"]["lr_scheduler"] = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience_validations": 0,
        "threshold": 1.0,
        "threshold_mode": "abs",
        "cooldown_validations": 0,
        "min_lr": 1e-6,
    }
    config["multi_object_training"].update(
        {"epochs": 3, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        output_dir=tmp_path,
        progress=True,
    )

    output = capsys.readouterr().out
    assert "lr=5.00000000e-03" in output
    assert "learning rate reduced: 1.00000000e-02 -> 5.00000000e-03" in output
    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert all("learning_rate" in item for item in history)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["initial_learning_rate"] == 0.01
    assert metrics["final_learning_rate"] == pytest.approx(0.0025)
    assert metrics["lr_scheduler_type"] == "reduce_on_plateau"
    assert metrics["lr_reduction_count"] == 2
    assert result.lr_reduction_count == 2


def test_max_optimizer_steps_can_stop_mid_epoch():
    config = _multi_config()
    config["multi_object_training"].update(
        {
            "epochs": 10,
            "gradient_accumulation_meshes": 1,
            "max_optimizer_steps": 3,
        }
    )

    result = train_multi_object(
        [_triangle_sample("train_a"), _triangle_sample("train_b")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert result.optimizer_steps == 3
    assert result.completed_epochs == 2
    assert result.stop_reason == "max_optimizer_steps"
    assert result.history[-1]["optimizer_steps_this_epoch"] == 1


def test_optimizer_step_reporting_writes_live_history(tmp_path, capsys):
    config = _multi_config()
    config["multi_object_training"].update(
        {
            "epochs": 10,
            "gradient_accumulation_meshes": 1,
            "max_optimizer_steps": 4,
            "report_every_optimizer_steps": 2,
        }
    )

    result = train_multi_object(
        [_triangle_sample("train_a"), _triangle_sample("train_b")],
        [_triangle_sample("validation")],
        config,
        output_dir=tmp_path,
        progress=True,
    )

    assert result.optimizer_steps == 4
    step_history = json.loads(
        (tmp_path / "training_step_history.json").read_text(encoding="utf-8")
    )
    assert [item["optimizer_steps"] for item in step_history] == [2, 4]
    assert [item["progress_percent"] for item in step_history] == [50.0, 100.0]
    assert all(math.isfinite(item["train_loss"]) for item in step_history)
    assert all(item["interval_seconds"] > 0 for item in step_history)
    assert all(item["validation_loss"] is not None for item in step_history)
    output = capsys.readouterr().out
    assert "step progress progress=50.00% step=2" in output
    assert "step progress progress=100.00% step=4" in output
    assert "epoch=" not in output


def test_early_stopping_counts_validation_events(monkeypatch):
    _use_constant_validation(monkeypatch, loss=1.0)
    config = _multi_config()
    config["multi_object_training"].update(
        {
            "epochs": 10,
            "validation_every_epochs": 1,
            "early_stopping": {
                "enabled": True,
                "patience_validations": 2,
                "min_delta": 0.0,
            },
        }
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert result.completed_epochs == 3
    assert result.stopped_early
    assert result.stop_reason == "early_stopping"


def test_default_view_selection_uses_all_views_and_pruned_worker_sample():
    prepared = _prepare_object_static(_multi_view_sample(), _multi_config())
    selected = _select_prepared_views(prepared, None, base_seed=11, epoch=1)
    item = _MaterializedPreparedDataset(
        [selected],
        decode_images=False,
        views_per_sample=None,
        base_seed=11,
        profile_loading=False,
    )[0]

    assert item["sample"]["images"].shape[0] == 14
    assert item["used_view_count"] == 14
    assert torch.equal(item["raw_target"], prepared.raw_target)
    assert item["face_count"] == prepared.face_count
    reconstructed = multi_trainer._prepared_from_loader_item(item)
    assert torch.equal(reconstructed.raw_target, prepared.raw_target)
    assert reconstructed.face_count == prepared.face_count
    for removed in (
        "gt_vertices",
        "gt_faces",
        "target_positions",
        "laplacian_target",
        "raw_laplacian_target",
        "normalized_laplacian_target",
        "local_edge_scale",
        "faces",
        "metadata",
    ):
        assert removed not in item["sample"]
    assert prepared.raw_target is not None
    assert prepared.face_count == 4


def test_epoch_aware_view_selection_is_aligned_and_reproducible():
    prepared = _prepare_object_static(_multi_view_sample(), _multi_config())
    first = _select_prepared_views(prepared, 4, base_seed=19, epoch=3)
    repeated = _select_prepared_views(prepared, 4, base_seed=19, epoch=3)
    next_epoch = _select_prepared_views(prepared, 4, base_seed=19, epoch=4)

    for name in ("images", "intrinsics", "extrinsics", "visibility"):
        assert first.sample[name].shape[0] == 4
        torch.testing.assert_close(first.sample[name], repeated.sample[name])
    assert len(first.sample["image_paths"]) == 4
    assert first.sample["image_paths"] == repeated.sample["image_paths"]
    selected_ids = first.sample["intrinsics"][:, 0, 1].to(dtype=torch.long)
    assert first.sample["image_paths"] == [f"view_{index:02d}.png" for index in selected_ids]
    torch.testing.assert_close(
        first.sample["extrinsics"][:, 0, 3], selected_ids.to(dtype=torch.float32)
    )
    torch.testing.assert_close(
        first.sample["images"][:, 0, 0, 0], selected_ids.to(dtype=torch.float32) / 20.0
    )
    expected_visibility = torch.tensor(
        [[(int(index) >> bit) & 1 for bit in range(4)] for index in selected_ids],
        dtype=torch.bool,
    )
    assert torch.equal(first.sample["visibility"], expected_visibility)
    assert first.sample["image_paths"] != next_epoch.sample["image_paths"]


@pytest.mark.parametrize(
    ("condition", "field"),
    [
        ("backface_only", "visibility_backface_only"),
        ("occlusion_only", "visibility_occlusion_only"),
        ("backface_and_occlusion", "visibility_backface_and_occlusion"),
    ],
)
def test_precomputed_renderer_visibility_condition_is_selected(condition, field):
    sample = _multi_view_sample(3)
    sample["visibility_backface_only"] = torch.tensor(
        [[True, False, True, False], [False, True, False, True], [True] * 4]
    )
    sample["visibility_occlusion_only"] = ~sample["visibility_backface_only"]
    sample["visibility_backface_and_occlusion"] = torch.zeros((3, 4), dtype=torch.bool)
    config = _multi_config()
    config["renderer_visibility"] = {"condition": condition}
    prepared = _prepare_object_static(sample, config)
    assert torch.equal(prepared.sample["visibility"], sample[field])
    assert "depth_maps" not in prepared.sample
    assert "depth_paths" not in prepared.sample


def test_frustum_only_explicitly_ignores_prepared_renderer_visibility():
    sample = _multi_view_sample(3)
    config = _multi_config()
    config["renderer_visibility"] = {"condition": "frustum_only"}
    prepared = _prepare_object_static(sample, config)
    assert prepared.sample["visibility"] is None


@pytest.mark.parametrize(
    ("input_mode", "zero_images"),
    [("coarse_only", False), ("coarse_plus_multiview", True)],
)
def test_image_free_ablation_does_not_open_lazy_images(
    tmp_path, monkeypatch, input_mode, zero_images
):
    sample = _triangle_sample("image_free")
    sample.pop("images")
    sample["image_paths"] = ["missing.png"]
    sample["source_image_size"] = [1920, 1920]
    sample["prepared_image_size"] = 1920
    sample["prepared_storage_format"] = "lazy_image_paths_v1"
    sample["_dataset_root"] = str(tmp_path)
    config = _multi_config()
    config["input_mode"] = input_mode
    config["data_loading"] = {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    config["multi_object_training"]["epochs"] = 1

    def forbidden_open(*args, **kwargs):
        raise AssertionError("Image.open must not run for image-free ablations")

    monkeypatch.setattr("mlr.learned_laplacian.sample_io.Image.open", forbidden_open)
    result = train_multi_object(
        [sample], None, config, zero_images=zero_images, progress=False
    )

    assert math.isfinite(result.final_train_loss)
    assert result.history[0]["decoded_image_bytes"] == 0
    assert result.history[0]["mean_used_view_count"] == 1


def test_manifest_training_uses_static_lazy_load_and_uint8_images(
    tmp_path, monkeypatch
):
    root = tmp_path / "dataset"
    prepared_dir = root / "prepared"
    image_dir = root / "images"
    prepared_dir.mkdir(parents=True)
    image_dir.mkdir()
    image_path = image_dir / "view.png"
    from PIL import Image

    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_path)

    def write_lazy_sample(sample_id, split):
        sample = _triangle_sample(sample_id)
        sample.pop("images")
        sample["image_paths"] = ["images/view.png"]
        sample["source_image_size"] = [16, 16]
        sample["prepared_image_size"] = 16
        sample["prepared_storage_format"] = "lazy_image_paths_v1"
        path = prepared_dir / f"{sample_id}.pt"
        save_prepared_sample(sample, path)
        return {"sample_id": sample_id, "path": f"prepared/{path.name}", "split": split}

    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    write_lazy_sample("train", "train"),
                    write_lazy_sample("validation", "validation"),
                ]
            }
        ),
        encoding="utf-8",
    )
    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")
    seen_image_dtypes = []
    moved_image_ranges = []
    original_move = multi_trainer._move_prepared_object_to_device

    def checked_move(prepared, *args, **kwargs):
        seen_image_dtypes.append(prepared.sample["images"].dtype)
        moved = original_move(prepared, *args, **kwargs)
        moved_image_ranges.append(
            (
                moved.sample["images"].dtype,
                float(moved.sample["images"].min()),
                float(moved.sample["images"].max()),
            )
        )
        return moved

    def forbidden_eager_getitem(self, index):
        raise AssertionError("PreparedMeshDataset.__getitem__ eagerly materialized images")

    monkeypatch.setattr(multi_trainer, "_move_prepared_object_to_device", checked_move)
    monkeypatch.setattr(PreparedMeshDataset, "__getitem__", forbidden_eager_getitem)
    config = _multi_config()
    config["input_mode"] = "coarse_plus_multiview"
    config["data_loading"] = {
        "num_workers": 1,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    config["multi_object_training"]["epochs"] = 1
    config["multi_object_training"]["cache_prepared_samples_on_device"] = False
    config["multi_object_training"]["profile_training"] = True

    result = train_multi_object(
        train_dataset,
        validation_dataset,
        config,
        progress=False,
    )

    assert math.isfinite(result.final_train_loss)
    assert seen_image_dtypes
    assert set(seen_image_dtypes) == {torch.uint8}
    assert moved_image_ranges
    assert all(dtype == torch.float32 for dtype, _, _ in moved_image_ranges)
    assert all(0.0 <= minimum <= maximum <= 1.0 for _, minimum, maximum in moved_image_ranges)
    assert result.history[0]["image_decode_resize_seconds"] >= 0.0
    assert result.history[0]["mean_used_view_count"] == 1.0
    assert result.history[0]["decoded_image_bytes"] == 3 * 16 * 16
