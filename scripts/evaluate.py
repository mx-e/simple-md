#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data /home/maxi/MOLECULAR_ML/5_refactored_repo/container.sif python
from loguru import logger
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from torch import multiprocessing as mp
import itertools

from pprint import pformat
from hydra_zen import instantiate, load_from_yaml, builds
import matplotlib.pyplot as plt
import torch.nn as nn

from pathlib import Path
import torch as th
import yaml, os

from lib.types import Property as Props, Split, DatasetSplits
from lib.utils.helpers import export_xyz
from lib.datasets import get_qcml_dataset
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.dist import setup_device
from lib.utils.run import run
from conf.base_conf import BaseConfig, configure_main
from lib.utils.checkpoint import load_checkpoint
from lib.data.loaders import collate_fn
from lib.data.transforms import center_positions_on_centroid, augment_positions

pbuilds = partial(builds, zen_partial=True)

qcml_data = pbuilds(
    get_qcml_dataset,
    data_dir="/home/maxi/MOLECULAR_ML/5_refactored_repo/data_ar",
    dataset_name="qcml_unified_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=True,
)


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,
    model_run_dir: Path,
    checkpoint_name: str = "best_model",
    data=qcml_data,
    batch_size: int = 1024,
    n_random_transforms: int = 128,
    n_samples: int = 10000,
    reflections: bool = False,
    rotations: bool = True,
    log_best_worst_conformers: int = 15,
    log_best_worst: int = 15,
    log_random: int = 15,
    log_uncertainty_performance_correlation: bool = True,
):
    mp.set_start_method("spawn", force=True)
    ctx, device = setup_device()
    config_path = model_run_dir / ".hydra" / "config.yaml"
    conf = load_from_yaml(config_path)
    model_conf = conf["train"]["model"]
    model = instantiate(model_conf)
    loss_module_conf = conf["train"]["loss"]
    loss_module = instantiate(loss_module_conf)
    loss_module.losses_per_atom = True

    job_dir = get_hydra_output_dir()
    out_dir = job_dir / "eval_results"
    out_dir.mkdir(exist_ok=True)

    checkpoint_path = Path(model_run_dir) / "ckpts" / (checkpoint_name + ".pth")
    load_checkpoint(model, checkpoint_path)
    predictor = Predictor(model, loss_module).eval().to(device)
    eval_results = {}
    # data
    data = data(rank=0, copy_to_temp=False)
    # evaluation

    equivariance_results = evaluate_equivariance(
        predictor,
        data,
        job_dir,
        device,
        ctx,
        batch_size=batch_size,
        n_random_transforms=n_random_transforms,
        n_samples=n_samples,
        reflections=reflections,
        rotations=rotations,
        log_best_worst_conformers=log_best_worst_conformers,
        log_best_worst=log_best_worst,
        log_random=log_random,
        log_uncertainty_performance_correlation=log_uncertainty_performance_correlation,
    )
    logger.info(f"Equivariance evaluation results:\n{pformat(equivariance_results)}")
    eval_results["equivariance"] = equivariance_results

    results_path = Path(job_dir) / "eval_results" / "results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(eval_results, f)
    logger.info(f"Results saved to {results_path}")


@th.no_grad()
def evaluate_equivariance(
    model: nn.Module,
    data: DatasetSplits,
    job_dir: Path,
    device,
    ctx,
    batch_size: int,
    n_random_transforms: int,
    n_samples: int,
    reflections: bool,
    rotations: bool,
    log_best_worst_conformers: int,
    log_best_worst: int,
    log_random: int,
    log_uncertainty_performance_correlation: bool,
):
    assert (
        batch_size % n_random_transforms == 0
    ), "batch_size must be divisible by n_random_transforms"
    batch_size = batch_size // n_random_transforms
    prebatch_preprocessors = [center_positions_on_centroid]
    postbatch_preprocessors = [
        partial(
            augment_positions,
            augmentation_mult=n_random_transforms,
            random_reflection=reflections,
            random_rotation=rotations,
        )
    ]
    loader = DataLoader(
        data.splits[Split.test],
        batch_size=batch_size,
        collate_fn=partial(
            collate_fn,
            props=data.dataset_props,
            device=device,
            pre_batch_preprocessors=prebatch_preprocessors,
            post_batch_preprocessors=postbatch_preprocessors,
        ),
        shuffle=True,
    )
    steps = n_samples // batch_size
    logger.info(f"Evaluating equivariance on {steps * batch_size} samples")
    conformer_force_losses = []
    conformer_data = []
    for batch in tqdm(itertools.islice(loader, steps), total=steps):
        with ctx:
            _, losses = model(batch)
        conformer_data.append({k: v[::n_random_transforms] for k, v in batch.items()})
        conformer_force_losses.append(
            losses[Props.forces].view(-1, n_random_transforms)
        )
    conformer_force_losses = th.cat(
        conformer_force_losses, dim=0
    )  # (n, n_random_transforms)
    conformer_data = [
        {k: v[i] for k, v in conf_data.items()}
        for conf_data in conformer_data
        for i in range(batch_size)
    ]

    conformer_force_losses_mean = conformer_force_losses.mean(dim=1)
    conformer_force_losses_std = conformer_force_losses_mean.var().sqrt()
    equivariance_std = conformer_force_losses.var(dim=1).sqrt()
    equivariance_loss_share = (equivariance_std / conformer_force_losses_mean).mean()
    equivariance_results = {
        "conformer_force_loss_mean": conformer_force_losses_mean.mean().item(),
        "conformer_force_loss_std": conformer_force_losses_std.item(),
        "equivariance_std": equivariance_std.mean().item(),
        "equivariance_loss_share": equivariance_loss_share.item(),
    }

    if log_uncertainty_performance_correlation:
        # plot x-axis: conformer_force_losses_mean, y-axis: equivariance_std
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            conformer_force_losses_mean.cpu(), equivariance_std.cpu(), alpha=0.2, s=5
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Conformer force loss mean")
        ax.set_ylabel("Equivariance std")
        plt.tight_layout()
        plt.savefig(Path(job_dir) / "eval_results" / "uncertainty_correlation.png")

    if log_best_worst_conformers > 0:
        sorted_args = equivariance_std.argsort()
        best_conformers_idx = sorted_args[:log_best_worst_conformers]
        worst_conformers_idx = sorted_args[-log_best_worst_conformers:]
        best_conformer_data = [conformer_data[i] for i in best_conformers_idx]
        worst_conformer_data = [conformer_data[i] for i in worst_conformers_idx]

        ## save conformers

        export_dir = Path(job_dir) / "eval_results" / "best_worst_equivariance"
        os.makedirs(export_dir, exist_ok=True)
        for i, data in enumerate(best_conformer_data):
            export_xyz(
                data,
                export_dir,
                f"best_conformer_{i}",
            )
            plot_invariance(model, data, ctx, export_dir, f"best_conformer_{i}")

        for i, data in enumerate(worst_conformer_data):
            export_xyz(
                data,
                export_dir,
                f"worst_conformer_{i}",
            )
            plot_invariance(model, data, ctx, export_dir, f"worst_conformer_{i}")
    if log_best_worst > 0:
        sorted_args = conformer_force_losses_mean.argsort()
        best_conformers_idx = sorted_args[:log_best_worst]
        worst_conformers_idx = sorted_args[-log_best_worst:]
        best_conformer_data = [conformer_data[i] for i in best_conformers_idx]
        worst_conformer_data = [conformer_data[i] for i in worst_conformers_idx]

        ## save conformers
        export_dir = Path(job_dir) / "eval_results" / "best_worst"
        os.makedirs(export_dir, exist_ok=True)
        for i, data in enumerate(best_conformer_data):
            export_xyz(
                data,
                export_dir,
                f"best_conformer_{i}",
            )
            plot_invariance(model, data, ctx, export_dir, f"best_conformer_{i}")
        for i, data in enumerate(worst_conformer_data):
            export_xyz(
                data,
                export_dir,
                f"worst_conformer_{i}",
            )
            plot_invariance(model, data, ctx, export_dir, f"worst_conformer_{i}")
    if log_random > 0:
        random_idx = th.randint(0, len(conformer_data), (log_random,))
        random_conformer_data = [conformer_data[i] for i in random_idx]
        export_dir = Path(job_dir) / "eval_results" / "random"
        os.makedirs(export_dir, exist_ok=True)
        for i, data in enumerate(random_conformer_data):
            export_xyz(
                data,
                export_dir,
                f"random_conformer_{i}",
            )
            plot_invariance(model, data, ctx, export_dir, f"random_conformer_{i}")

    return equivariance_results


class Predictor(nn.Module):
    def __init__(self, encoder, loss_module):
        super().__init__()
        self.encoder = encoder
        self.loss_module = loss_module

    def forward(self, inputs):
        out = self.encoder(inputs)
        loss = self.loss_module(out, inputs)
        return out, loss


def get_rotation_from_axis_angle(axis, angle):
    assert axis.shape[-1] == 3, "Axis should have shape (batch_dim, 3)"
    assert axis.shape[0] == angle.shape[0], "Batch dimensions should match"
    batch_dim = axis.shape[0]
    axis = axis / th.norm(axis, dim=-1, keepdim=True)  # (n, 3)

    cos_theta = th.cos(angle)  # (n,)
    sin_theta = th.sin(angle)  # (n,)

    cos_theta = cos_theta.view(batch_dim, 1, 1)  # (n, 1, 1)
    sin_theta = sin_theta.view(batch_dim, 1, 1)  # (n, 1, 1)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]  # (n,)

    K = th.stack(
        [
            th.stack([th.zeros_like(x), -z, y], dim=-1),
            th.stack([z, th.zeros_like(x), -x], dim=-1),
            th.stack([-y, x, th.zeros_like(x)], dim=-1),
        ],
        dim=-2,
    )  # (n, 3, 3)
    outer_product = th.bmm(axis.unsqueeze(2), axis.unsqueeze(1))  # (n, 3, 3)
    identity = (
        th.eye(3, device=axis.device).unsqueeze(0).expand(batch_dim, -1, -1)
    )  # (n, 3, 3)
    rotation_matrices = (
        cos_theta * identity + sin_theta * K + (1 - cos_theta) * outer_product
    )  # (n, 3, 3)
    return rotation_matrices


@th.no_grad()
def plot_invariance(model, data, ctx, export_dir, export_name):
    axes = th.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        device=data[Props.positions].device,
        dtype=th.float32,
    )
    n_interploations = 120
    angles = th.linspace(
        0, 2 * th.pi, n_interploations, device=data[Props.positions].device
    )  # (n_interploations,)
    axes = axes.repeat_interleave(n_interploations, dim=0)  # (3 * n_interploations, 3)
    angles = angles.repeat(3)  # (3 * n_interploations,)
    R = get_rotation_from_axis_angle(axes, angles)  # (3 * n_interploations, 3, 3)
    data = {
        k: v.unsqueeze(0).repeat(n_interploations * 3, *(len(v.shape) * [1]))
        for k, v in data.items()
    }
    data[Props.positions] = th.einsum(
        "bij,bnj->bni", R, data[Props.positions]
    )  # (3 * n_interploations, s, 3)
    data[Props.forces] = th.einsum("bij,bnj->bni", R, data[Props.forces])
    with ctx:
        _, losses = model(data)

    loss_x, loss_y, loss_z = (
        losses[Props.forces].cpu().chunk(3, dim=0)
    )  # (n_interploations,) * 3
    angles = angles.cpu().numpy()

    export_path = export_dir / f"{export_name}_rotation_loss.png"
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title("Force loss invariance along rotation axes")
    ax.plot(angles[:n_interploations], loss_x, label="x", color="r", linestyle="-")
    ax.plot(angles[:n_interploations], loss_y, label="y", color="g", linestyle="--")
    ax.plot(angles[:n_interploations], loss_z, label="z", color="b", linestyle="-.")
    ax.set_xlabel("Rotation angle (rad)")
    ax.set_ylabel("Force loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(export_path)
    plt.close(fig)
    return export_path


if __name__ == "__main__":
    run(main)
