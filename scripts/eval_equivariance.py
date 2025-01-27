#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif python

from functools import partial
from itertools import islice
from pathlib import Path
from typing import Literal

import torch as th
import torch.multiprocessing as mp
import yaml
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import builds, instantiate, load_from_yaml
from lib.data.loaders import get_loaders
from lib.data.transforms import augment_positions
from lib.datasets import get_qcml_dataset
from lib.loss import LossModule
from lib.models.pair_encoder import get_pair_encoder_pipeline_config
from lib.train_loop import Predictor
from lib.types import Property as Props
from lib.types import Split
from lib.utils.augmentation import (
    analyze_rotation_distribution,
    generate_equidistant_rotations,
    visualize_rotations,
)
from lib.utils.checkpoint import load_checkpoint
from lib.utils.dist import get_amp, setup_device
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from loguru import logger
from torch.nn import functional as F

pbuilds = partial(builds, zen_partial=True)

qcml_data = builds(
    get_qcml_dataset,
    rank=0,
    data_dir="./data_ar",
    dataset_name="qcml_unified_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=False,
)

loss_module = builds(
    LossModule,
    targets=["forces"],
    loss_types={"forces": "euclidean"},
    metrics={"forces": ["mae", "mse", "euclidean"]},
    losses_per_mol=True,
)

pair_encoder_data_config = builds(
    get_pair_encoder_pipeline_config,
    augmentation_mult=1,
    random_rotation=False,
    random_reflection=False,
    center_positions=True,
    dynamic_batch_size_cutoff=10000,
    include_dipole=False,
)


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,
    model_run_dir: Path,
    dataset=qcml_data,
    checkpoint_name: str = "best_model",
    ptdtype: Literal["float32", "bfloat16", "float16"] = "float32",
    loss_module=loss_module,
    pipeline_config=pair_encoder_data_config,
    n_test_samples: int = 5,
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
    model = Predictor(model, loss_module).eval().to(device)

    loaders = get_loaders(
        rank=0,
        batch_size=1,
        grad_accum_steps=1,
        world_size=1,
        device=device,
        dataset_splits=dataset,
        pipeline_config=pipeline_config,
        num_workers=0,
    )

    out_dir = job_dir / "eval_equivariance"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluate_equivariance(output_dir=out_dir, model=model, loaders=loaders, n_test_samples=n_test_samples)


def evaluate_equivariance(output_dir: Path, model, loaders, n_test_samples) -> None:
    logger.info("Evaluating equivariance of the model.")
    std_deviations = {Split.train: {}, Split.test: {}, Split.val: {}}

    for split, loader in loaders.items():
        for n_averaging_rotations in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if n_averaging_rotations > 1:
                logger.info(f"Generating {n_averaging_rotations} equidistant rotation to average model predictions.")
                equidistant_rotations = generate_equidistant_rotations(n_averaging_rotations)
                logger.info(analyze_rotation_distribution(equidistant_rotations))
            else:
                equidistant_rotations = th.eye(3).unsqueeze(0)
            std_dev = compute_equivariance(model, loader, equidistant_rotations, n_samples=n_test_samples)
            std_deviations[split][n_averaging_rotations] = std_dev

    logger.info(f"Results: {std_deviations}")
    # save as yaml
    out_path = output_dir / "equivariance_results.yaml"
    with out_path.open("w") as f:
        yaml.dump(std_deviations, f)
    logger.info(f"Results saved in {out_path}")
    return std_deviations


@th.no_grad()
def compute_equivariance(
    model,
    loader,
    rotations_model: th.Tensor,  # (n_rotations, 3, 3)
    n_rotations_eval: int = 512,
    n_samples: int = 256,
) -> None:
    n_model_rots, _, _ = rotations_model.shape

    stds = []

    for batch in islice(loader, n_samples):
        augmented_batch = augment_positions(
            batch, augmentation_mult=n_rotations_eval, random_rotation=True, random_reflection=False
        )
        augmented_batch_model, reverse_rotations = apply_rotation_matrices_to_batch(
            augmented_batch, rotations_model
        )  # _ , (b * n_rotations, 3, 3)
        SUB_BATCH_SIZE = 1024
        total_bs = n_model_rots * n_rotations_eval
        sub_rot_eval = (
            n_rotations_eval if total_bs < SUB_BATCH_SIZE else n_rotations_eval / (total_bs / SUB_BATCH_SIZE)
        )
        assert sub_rot_eval.is_integer(), f"model rotations * eval rotations must be divisible by {SUB_BATCH_SIZE}"
        assert SUB_BATCH_SIZE % n_model_rots == 0, f"model rotations must divide {SUB_BATCH_SIZE}"
        assert n_model_rots < SUB_BATCH_SIZE, f"model rotations must be less than {SUB_BATCH_SIZE}"

        losses_mol = []
        # to little memory to evaluate all rotations at once
        for i in range(0, total_bs, SUB_BATCH_SIZE):
            sub_batch = {k: v[i : i + SUB_BATCH_SIZE] for k, v in augmented_batch_model.items()}
            sub_batch_reverse_rots = reverse_rotations[i : i + SUB_BATCH_SIZE]
            loss = get_batch_equivariance_losses(
                sub_batch, model, n_model_rots, sub_batch_reverse_rots, int(sub_rot_eval)
            )
            losses_mol.append(loss)
        stds_mol = th.cat(losses_mol, dim=0).std().item()
        stds.append(stds_mol)

    stds = th.tensor(stds)
    mean_std_deviation = stds.mean().item()
    logger.info(
        f"Mean standard deviation of equivariance with averaging over {n_model_rots} rotations: {mean_std_deviation}"
    )
    return mean_std_deviation


def apply_rotation_matrices_to_batch(batch: dict, rotation_matrices: th.Tensor) -> dict:
    positions, forces = batch[Props.positions], batch[Props.forces]  # (b, n_atoms, 3), (b, n_atoms, 3)
    n_rotations, _, _ = rotation_matrices.shape
    b, n_atoms, _ = positions.shape

    positions = positions.repeat_interleave(n_rotations, dim=0)  # (b * n_rotations, n_atoms, 3)
    forces = forces.repeat_interleave(n_rotations, dim=0)  # (b * n_rotations, n_atoms, 3)

    R = rotation_matrices.repeat(b, 1, 1).to(th.float).to(positions.device)  # (b * n_rotations, 3, 3)

    positions = th.bmm(positions, R)  # (b * n_rotations, n_atoms, 3)
    forces = th.bmm(forces, R)  # (b * n_rotations, n_atoms, 3)

    reverse_rotations = R.transpose(1, 2)  # (b * n_rotations, 3, 3)

    batch[Props.positions] = positions
    batch[Props.forces] = forces
    batch[Props.mask] = batch[Props.mask].repeat_interleave(n_rotations, dim=0)
    batch[Props.atomic_numbers] = batch[Props.atomic_numbers].repeat_interleave(n_rotations, dim=0)
    batch[Props.multiplicity] = batch[Props.multiplicity].repeat_interleave(n_rotations, dim=0)
    batch[Props.charge] = batch[Props.charge].repeat_interleave(n_rotations, dim=0)
    return batch, reverse_rotations


def get_batch_equivariance_losses(
    batch,
    model,
    n_model_rots: int,
    reverse_rotations: th.Tensor,
    n_rotations_eval: int = 512,
) -> None:
    out, _ = model(batch)
    assert th.all(batch[Props.mask] == 1), "This eval does not support ragged batches"

    forces_true, forces_pred = batch[Props.forces], out[Props.forces]  # (b, n, 3), (b, n, 3)

    # reverse frame averaging rotations
    forces_true = th.bmm(forces_true, reverse_rotations)  # (b * n_rotations, n, 3)
    forces_pred = th.bmm(forces_pred, reverse_rotations)  # (b * n_rotations, n, 3)

    # average over frame averaging rotations
    forces_true = forces_true.view(n_rotations_eval, n_model_rots, -1, 3)  # (n_rotations_eval, n_model_rots, n, 3)
    forces_pred = forces_pred.view(n_rotations_eval, n_model_rots, -1, 3)  # (n_rotations_eval, n_model_rots, n, 3)

    forces_true = forces_true.mean(dim=1)  # (n_rotations_eval, n, 3)
    forces_pred = forces_pred.mean(dim=1)  # (n_rotations_eval, n, 3)

    # std of loss over evaluation rotations
    dist = (forces_true - forces_pred).norm(dim=-1)  # (n_rotations_eval, n)
    loss = dist.mean(dim=-1)  # (n_rotations_eval,)
    return loss


if __name__ == "__main__":
    run(main)
