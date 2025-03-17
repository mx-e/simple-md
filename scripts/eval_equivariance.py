#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif uv run python

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
from lib.datasets import get_qcml_dataset
from lib.loss import LossModule
from lib.models.pair_encoder import get_pair_encoder_pipeline_config
from lib.train_loop import Predictor
from lib.types import Property as Props
from lib.types import Split
from lib.utils.augmentation import (
    analyze_rotation_distribution,
    generate_equidistant_rotations,
    get_random_rotations,
)
from lib.utils.checkpoint import load_checkpoint
from lib.utils.dist import get_amp, setup_device
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from loguru import logger
from ase.io import read
from lib.types import property_dtype

pbuilds = partial(builds, zen_partial=True)

qcml_data = builds(
    get_qcml_dataset,
    rank=0,
    data_dir="./data_ar",
    dataset_name="qcml_unified_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=False,
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

ANG_TO_BOHR = 1.0 / 0.529177249


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,
    model_run_dir: Path,
    dataset=qcml_data,
    xyz_structs: list[str] | None = [
        "md17_ethanol",
        "md17_benzene",
        "md17_aspirin",
        "md17_malonaldehyde",
        "md17_naphthalene",
        "md17_salicylic",
        "md17_toluene",
        "md17_uracil",
        "c1h4",
        "c2h2",
        "c2h4",
        "c2h6",
        "c3h4",
        "c3h8",
        "c4h4",
        "c4h10",
        "c5h4",
        "c5h12",
        "c6h4",
        "c6h14",
        "c7h4",
        "c7h16",
        "c8h4",
        "c8h18",
        "c9h4",
        "c9h20",
        "c10h4",
        "c10h22",
        "c11h4",
        "c11h24",
        "c12h4",
        "c12h26",
        "c13h4",
        "c13h28",
        "c14h4",
        "c14h30",
        "c15h4",
        "c15h32",
        "c16h4",
        "c16h34",
    ],  # noqa: B006
    checkpoint_name: str = "best_model",
    pipeline_config=pair_encoder_data_config,
    n_test_samples: int = 2048,
    init_struct_dir: Path = "data_md",
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    job_dir = get_hydra_output_dir()
    device = setup_device()
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
    model = model.eval().to(device)

    out_dir = job_dir / "eval_equivariance"
    out_dir.mkdir(parents=True, exist_ok=True)
    if xyz_structs is not None:
        logger.info(
            f"XYZ structs provided, evaluating equivariance for {len(xyz_structs)} structures instead of loading dataset."
        )
        valid_init_struct_names = [f.stem for f in Path(init_struct_dir).glob("*.xyz")]
        for struct_name in xyz_structs:
            assert struct_name in valid_init_struct_names, (
                f"Provided structure {struct_name} not found in {init_struct_dir}"
            )
        evaluate_equivariance_structures(
            output_dir=out_dir,
            model=model,
            struct_dir=init_struct_dir,
            struct_names=xyz_structs,
        )
    else:
        logger.info("Evaluating on dataset samples, for custom structures provide xyz_structs.")
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
        evaluate_equivariance_dataset(output_dir=out_dir, model=model, loaders=loaders, n_test_samples=n_test_samples)


def evaluate_equivariance_structures(output_dir: Path, model, struct_dir: Path, struct_names: list[str]) -> None:
    logger.info(f"Evaluating equivariance of the model on custom structures: {struct_names}")
    std_deviations = {}

    for struct_name in struct_names:
        logger.info(f"Processing structure: {struct_name}")
        xyz_path = struct_dir / f"{struct_name}.xyz"

        device = model.parameters().__next__().device  # load struct to same device as model
        struct_data = load_xyz_structure(xyz_path, device=device)
        struct_loader = [struct_data]  # Single-item loader

        struct_deviations = {}
        for n_averaging_rotations in [1, 2, 4, 8, 16, 32, 64, 128]:
            if n_averaging_rotations > 1:
                logger.info(f"Generating {n_averaging_rotations} equidistant rotations to average model predictions.")
                equidistant_rotations = generate_equidistant_rotations(n_averaging_rotations)
                logger.info(analyze_rotation_distribution(equidistant_rotations))
            else:
                equidistant_rotations = th.eye(3).unsqueeze(0)

            std_dev = compute_equivariance(
                model=model,
                loader=struct_loader,
                fa_rotation_matrices=equidistant_rotations,
                n_samples=1,  # Single structure
            )
            struct_deviations[n_averaging_rotations] = std_dev

        std_deviations[struct_name] = struct_deviations

    logger.info(f"Results: {std_deviations}")
    # Save as yaml
    out_path = output_dir / "equivariance_results_structures.yaml"
    with out_path.open("w") as f:
        yaml.dump(std_deviations, f)
    logger.info(f"Results saved in {out_path}")
    return std_deviations


def load_xyz_structure(xyz_path: Path, device) -> dict:
    atoms = read(str(xyz_path))
    positions = th.tensor(atoms.get_positions()).unsqueeze(0) * ANG_TO_BOHR
    atomic_numbers = th.tensor(atoms.get_atomic_numbers()).unsqueeze(0)
    mask = th.ones_like(atomic_numbers)
    charge = th.tensor([atoms.info.get("charge", 0)]).unsqueeze(0)
    multiplicity = th.tensor([atoms.info.get("multiplicity", 1)]).unsqueeze(0)

    batch = {
        Props.positions: positions,
        Props.atomic_numbers: atomic_numbers,
        Props.mask: mask,
        Props.charge: charge,
        Props.multiplicity: multiplicity,
    }
    batch = {k: v.to(property_dtype[k]).to(device) for k, v in batch.items()}
    return batch


def evaluate_equivariance_dataset(output_dir: Path, model, loaders, n_test_samples) -> None:
    logger.info("Evaluating equivariance of the model.")
    std_deviations = {str(Split.train): {}, str(Split.test): {}, str(Split.val): {}}

    for split, loader in loaders.items():
        for n_averaging_rotations in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if n_averaging_rotations > 1:
                logger.info(f"Generating {n_averaging_rotations} equidistant rotation to average model predictions.")
                equidistant_rotations = generate_equidistant_rotations(n_averaging_rotations)
                logger.info(analyze_rotation_distribution(equidistant_rotations))
            else:
                equidistant_rotations = th.eye(3).unsqueeze(0)
            std_dev = compute_equivariance(model, loader, equidistant_rotations, n_samples=n_test_samples)
            std_deviations[str(split)][n_averaging_rotations] = std_dev

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
    fa_rotation_matrices: th.Tensor,  # (n_rotations, 3, 3)
    n_rotations_eval: int = 512,
    n_samples: int = 256,
) -> None:
    n_model_rots, _, _ = fa_rotation_matrices.shape

    stds = []

    for mol in islice(loader, n_samples):
        mol_so3_sample, R_inv = augment_positions_with_SO3_sample(batch=mol, n_rotations_eval=n_rotations_eval)
        forces_mol = get_batch_force_predictions(
            batch=mol_so3_sample,
            model=model,
            n_frame_averaging=n_model_rots,
            fa_rotation_matrices=fa_rotation_matrices,
            n_rotations_eval=n_rotations_eval,
        )  # (n_rotations_eval, n_atoms, 3)

        forces_mol = th.bmm(forces_mol, R_inv)  # (n_rotations_eval, n_atoms, 3)
        forces_mol_mean = forces_mol.mean(dim=0, keepdim=True)  # (1, n_atoms, 3)
        dists = (forces_mol - forces_mol_mean).norm(dim=-1).mean(-1)  # (n_rotations_eval)
        stds.append(dists.mean().item())

    stds = th.tensor(stds)
    mean_std_deviation = stds.mean().item()
    logger.info(f"Mean distance from group mean with averaging over {n_model_rots} rotations: {mean_std_deviation}")
    return mean_std_deviation


def augment_positions_with_SO3_sample(  # noqa: N802
    batch: dict,
    n_rotations_eval: int,
) -> None:
    batch = batch.copy()
    # augment positions
    for k, v in batch.items():
        batch[k] = v.repeat_interleave(n_rotations_eval, dim=0)

    positions = batch[Props.positions].double()  # (n_rotations, n_atoms, 3)

    n_batches, _, _ = positions.size()
    R = get_random_rotations(n_batches, positions.device).double()  # (n_rotations, 3, 3)
    batch[Props.positions] = th.bmm(positions, R)  # (n_rotations, n_atoms, 3)

    R_inv = R.transpose(1, 2)  # (n_rotations, 3, 3)
    return batch, R_inv


def get_batch_force_predictions(
    batch,
    model,
    n_frame_averaging: int,
    fa_rotation_matrices: th.Tensor,
    n_rotations_eval: int = 512,
) -> th.Tensor:
    assert th.all(batch[Props.mask] == 1), "This eval does not support ragged batches"
    MAX_BATCH_SIZE = 1024  # we want to be as fast as possible but not run out of memory
    total_bs = n_frame_averaging * n_rotations_eval
    sub_batch_size = n_rotations_eval if total_bs < MAX_BATCH_SIZE else n_rotations_eval / (total_bs / MAX_BATCH_SIZE)
    assert sub_batch_size.is_integer(), (
        f"frame averaging rotations * eval rotations must be divisible by {MAX_BATCH_SIZE}"
    )
    sub_batch_size = int(sub_batch_size)
    assert MAX_BATCH_SIZE % n_frame_averaging == 0, f"frame averaging rotations must divide {MAX_BATCH_SIZE}"
    assert n_frame_averaging < MAX_BATCH_SIZE, f"frame averaging rotations must be less than {MAX_BATCH_SIZE}"

    forces_mol = []
    # to little memory to evaluate all rotations at once for large frame_averaging sizes
    for i in range(0, n_rotations_eval, sub_batch_size):
        sub_batch = {k: v[i : i + sub_batch_size] for k, v in batch.items()}
        forces_pred = forward_with_frame_averaging(
            sub_batch, model, n_frame_averaging, fa_rotation_matrices
        )  # (sub_batch_size, n_atoms, 3)
        forces_mol.append(forces_pred)

    forces_mol = th.cat(forces_mol)  # (n_rotations_eval, n_atoms, 3)
    return forces_mol


def forward_with_frame_averaging(
    batch: dict,
    model,
    n_frame_averaging: int,
    fa_rotation_matrices: th.Tensor,
) -> th.Tensor:
    batch_size = batch[Props.positions].shape[0]
    # apply frame averaging rotations
    batch_frame_avging, reverse_rotations = apply_rotation_matrices_to_batch(batch, fa_rotation_matrices)
    out = model(batch_frame_avging)
    forces_pred = out[Props.forces].double()  # (batch_size * fa_size, n_atoms, 3)
    forces_pred = th.bmm(forces_pred, reverse_rotations)  # (batch_size * fa_size, n_atoms, 3)
    forces_pred = forces_pred.view(batch_size, n_frame_averaging, -1, 3)  # (batch_size, fa_size, n_atoms, 3)
    # mean over frame averaging rotations
    forces_pred = forces_pred.mean(dim=1)  # (batch_size, n, 3)
    return forces_pred  # (batch_size, fa_size, n_atoms, 3), (batch_size, n_atoms, 3)


def apply_rotation_matrices_to_batch(batch: dict, rotation_matrices: th.Tensor) -> dict:
    positions = batch[Props.positions]  # (b, n_atoms, 3)
    n_rotations, _, _ = rotation_matrices.shape
    b, n_atoms, _ = positions.shape

    positions = positions.repeat_interleave(n_rotations, dim=0).double()  # (b * n_rotations, n_atoms, 3)

    R = rotation_matrices.repeat(b, 1, 1).to(positions.device).double()  # (b * n_rotations, 3, 3)

    positions = th.bmm(positions, R)  # (b * n_rotations, n_atoms, 3)

    reverse_rotations = R.transpose(1, 2)  # (b * n_rotations, 3, 3)

    batch = batch.copy()
    batch[Props.positions] = positions.float()  # (b * n_rotations, n_atoms, 3)
    batch[Props.mask] = batch[Props.mask].repeat_interleave(n_rotations, dim=0)  # (b * n_rotations, n_atoms)
    batch[Props.atomic_numbers] = batch[Props.atomic_numbers].repeat_interleave(
        n_rotations, dim=0
    )  # (b * n_rotation, n_atoms)
    batch[Props.multiplicity] = batch[Props.multiplicity].repeat_interleave(n_rotations, dim=0)  # (b * n_rotations)
    batch[Props.charge] = batch[Props.charge].repeat_interleave(n_rotations, dim=0)  # (b * n_rotations)
    return batch, reverse_rotations


if __name__ == "__main__":
    run(main)
