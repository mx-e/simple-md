#!/usr/bin/env python3

import argparse
from copy import copy
from pathlib import Path

import torch as th
import yaml
from loguru import logger

# Template for the new config format
NEW_CONFIG_TEMPLATE = {
    "_target_": "__main__.main",
    "cfg": {
        "_target_": "conf.base_conf.BaseConfig",
        "seed": 42,
        "runtime": {
            "_target_": "conf.base_conf.RuntimeInfo",
            "device": "cuda",
            "out_dir": None,
            "n_gpu": 1,
            "node_hostname": "head024",
        },
        "loglevel": "debug",
        "wandb": {
            "_target_": "lib.utils.wandb.WandBRun",
            "group": None,
            "mode": "online",
        },
        "job": None,
    },
    "train": {
        "_target_": "__main__.train",
        "_partial_": True,
        "model": {
            "_target_": "lib.models.pair_encoder.PairEncoder",
            "n_layers": 3,
            "embd_dim": 192,
            "num_3d_kernels": 128,
            "cls_token": False,
            "num_heads": 12,
            "activation": "gelu",
            "ffn_multiplier": 4,
            "attention_dropout": 0.0,
            "ffn_dropout": 0.0,
            "head_dropout": 0.0,
            "norm_first": True,
            "norm": "layer",
            "decomposer_type": "pooling",
            "target_heads": ["forces"],
            "head_project_down": True,
        },
        "data": {
            "_target_": "lib.datasets.qcml.get_qcml_dataset",
            "_partial_": True,
            "data_dir": "/home/maxi/MOLECULAR_ML/5_refactored_repo/data_ar",
            "dataset_name": "qcml_unified_fixed_split_by_smiles",
            "dataset_version": "1.0.0",
            "copy_to_temp": True,
        },
        "pipeline_conf": {
            "_target_": "lib.models.pair_encoder.get_pair_encoder_pipeline_config",
            "augmentation_mult": 2,
            "random_rotation": True,
            "random_reflection": True,
            "center_positions": True,
            "dynamic_batch_size_cutoff": 29,
        },
        "loss": {
            "_target_": "lib.loss.LossModule",
            "targets": ["forces"],
            "loss_types": {"forces": "euclidean"},
            "metrics": {"forces": ["mae", "mse", "rmse", "euclidean"]},
        },
        "lr_scheduler": {
            "_target_": "lib.lr_scheduler.get_lr_scheduler",
            "_partial_": True,
            "scheduler_type": "cosine_warmup",
            "warmup_steps": 15000,
            "min_lr": 1e-07,
        },
        "ema": {"_target_": "lib.ema.EMAModel", "_partial_": True, "decay": 0.9997},
        "optimizer": {
            "_target_": "torch.optim.adamw.AdamW",
            "_partial_": True,
            "weight_decay": 1e-07,
        },
        "batch_size": 256,
        "total_steps": 880000,
        "lr": 0.0005,
        "grad_accum_steps": 1,
        "log_interval": 5,
        "eval_interval": 100,
        "save_interval": 50000,
        "eval_samples": 50000,
        "clip_grad": 1.0,
        "checkpoint_path": None,
    },
}


def convert_checkpoint(path_old_checkpoint: Path, new_checkpoint_dir: Path) -> Path:
    """Convert a single checkpoint from the old format to the new format."""
    logger.info(f"Converting checkpoint: {path_old_checkpoint}")
    model_dict_old = th.load(path_old_checkpoint, map_location="cpu")
    new_state_dict = {
        "optimizer_state_dict": model_dict_old["optimizer_state_dict"],
        "step": model_dict_old["step"],
    }

    if model_dict_old.get("ema_state_dict") is not None:
        new_state_dict["ema_state_dict"] = model_dict_old["ema_state_dict"]

    new_model_state_dict = {}
    for k, v in model_dict_old["model_state_dict"].items():
        if k.startswith("model."):
            k_new = k.replace("model.", "")
            new_model_state_dict[k_new] = v
        elif k.startswith("embed."):
            k_new = k.replace("embed.", "embedding.", 1)
            new_model_state_dict[k_new] = v
        elif k.startswith("regr_head."):
            if "forces" in k:
                k_new = k.replace("regr_head.mlp_forces.", "heads.forces.mlp.")
            elif "energy" in k:
                raise NotImplementedError("Cannot convert models trained on energy loss")
            else:
                k_new = k.replace("regr_head.", "heads.forces.")
            new_model_state_dict[k_new] = v

    new_state_dict["model_state_dict"] = new_model_state_dict
    new_ckpt_path = new_checkpoint_dir / path_old_checkpoint.name
    th.save(new_state_dict, new_ckpt_path)
    logger.info(f"Saved converted checkpoint to: {new_ckpt_path}")
    return new_ckpt_path


def transfer_config_values(config_old, config_new) -> None:
    """Transfer values from old config format to new config format."""
    mdl_conf = config_new["train"]["model"]
    mdl_conf_old = config_old["model"]
    loss_conf = config_new["train"]["loss"]
    loss_conf_old = config_old["model"]["loss"]

    # Transfer model configuration
    model_params = [
        "n_layers",
        "embd_dim",
        "num_3d_kernels",
        "cls_token",
        "num_heads",
        "activation",
        "ffn_multiplier",
        "attention_dropout",
        "ffn_dropout",
        "head_dropout",
        "norm_first",
        "norm",
        "decomposer_type",
    ]

    for param in model_params:
        mdl_conf[param] = mdl_conf_old[param]

    mdl_conf["target_heads"] = copy(loss_conf_old["targets"])
    mdl_conf["head_project_down"] = mdl_conf_old["project_down"]

    # Transfer loss configuration
    loss_conf["targets"] = loss_conf_old["targets"]
    loss_conf["loss_types"] = {"forces": "mae"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert checkpoints from old to new format")
    parser.add_argument(
        "-i",
        "--old_checkpoint_dir",
        type=str,
        help="Path to directory containing old checkpoints",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to output directory for converted checkpoints",
        required=True,
    )
    args = parser.parse_args()

    # Convert paths to Path objects
    old_ckpt_dir = Path(args.old_checkpoint_dir)
    output_dir = Path(args.output_dir)

    # Verify input directory exists
    if not old_ckpt_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {old_ckpt_dir}")

    # Create output directory structure
    out_path_config = output_dir / ".hydra"
    out_path_config.mkdir(parents=True, exist_ok=True)
    out_path_ckpts = output_dir / "ckpts"
    out_path_ckpts.mkdir(parents=True, exist_ok=True)

    # Load old config
    old_config_path = old_ckpt_dir / ".hydra" / "config.yaml"
    if not old_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {old_config_path}")

    with old_config_path.open("r") as f:
        config_old = yaml.safe_load(f)

    # Create new config based on template
    config_new = copy(NEW_CONFIG_TEMPLATE)
    transfer_config_values(config_old, config_new)

    # Save configs
    config_path = out_path_config / "config.yaml"
    legacy_config_path = out_path_config / "config_legacy_format.yaml"

    # Save old config for reference
    with legacy_config_path.open("w") as f:
        yaml.dump(config_old, f)

    # Save new config with header comment
    with config_path.open("w") as f:
        f.write(
            "# This config was converted from a legacy format to be compatible with inference - "
            "only values needed for inference are accurate - for reproducing the training run "
            "see 'config_legacy_format.yaml' file.\n"
        )
        yaml.dump(config_new, f)

    # Convert checkpoints
    checkpoint_files = list(old_ckpt_dir.glob("ckpts/*.pth"))
    if not checkpoint_files:
        logger.info("Warning: No checkpoint files found in input directory")

    for ckpt_path in checkpoint_files:
        try:
            convert_checkpoint(ckpt_path, out_path_ckpts)
        except Exception as e:
            logger.error(f"Error converting checkpoint {ckpt_path}: {e!s}")

    logger.info(f"\nConversion completed. Converted checkpoints and configs saved to: {output_dir}")


if __name__ == "__main__":
    main()
