#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif python

from pathlib import Path
from typing import Literal

import torch.multiprocessing as mp
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import instantiate, load_from_yaml
from lib.utils.checkpoint import load_checkpoint
from lib.utils.dist import get_amp, setup_device
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from loguru import logger

from scripts.lib.utils.augmentation import (
    analyze_rotation_distribution,
    generate_equidistant_rotations,
    visualize_rotations,
)


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,
    model_run_dir: Path,
    checkpoint_name: str = "best_model",
    ptdtype: Literal["float32", "bfloat16", "float16"] = "float32",
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    job_dir = get_hydra_output_dir()
    device, ctx = setup_device(), get_amp(ptdtype)
    model_run_conf_path = model_run_dir / ".hydra" / "config.yaml"
    model_run_conf = load_from_yaml(model_run_conf_path)
    if "ft" in model_run_conf:
        logger.info("detected fine-tuned model, loading pretrain model configuration")
        model_pt_conf_path = model_run_dir / ".hydra" / "model_pretrain_conf.yaml"
        model_conf = load_from_yaml(model_pt_conf_path)
    elif "train" in model_run_conf:
        model_conf = model_run_conf["train"]["model"]
    model = instantiate(model_conf)
    checkpoint_path = Path(model_run_dir) / "ckpts" / (checkpoint_name + ".pth")
    load_checkpoint(model, checkpoint_path)
    model.eval().to(device)

    out_dir = job_dir / "eval_equivariance"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluate_equivariance(
        output_dir=out_dir,
    )


def evaluate_equivariance(
    output_dir: Path,
):
    logger.info("Evaluating equivariance of the model.")
    rotations = generate_equidistant_rotations(200)
    logger.info(analyze_rotation_distribution(rotations))
    visualize_rotations(rotations, output_dir)


if __name__ == "__main__":
    run(main)
