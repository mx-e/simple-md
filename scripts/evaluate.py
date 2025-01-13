#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif python
import itertools
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Literal

import matplotlib.pyplot as plt
import torch as th
import yaml
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import builds, instantiate, load_from_yaml, store
from lib.data.loaders import collate_fn
from lib.data.transforms import augment_positions, center_positions_on_centroid
from lib.datasets import get_qcml_dataset, get_md17_22_dataset
from lib.loss import LossModule
from lib.train_loop import Predictor
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from lib.utils.checkpoint import load_checkpoint
from lib.utils.dist import get_amp, setup_device
from lib.utils.helpers import export_xyz, get_hydra_output_dir
from lib.utils.run import run
from loguru import logger
from torch import multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import MISSING

pbuilds = partial(builds, zen_partial=True)

qcml_data = pbuilds(
    get_qcml_dataset,
    data_dir="data/data_arrecord",
    dataset_name="qcml_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=True,
)

md17_benzene = pbuilds(
    get_md17_22_dataset,
    data_dir="./data",
    molecule_name="benzene",
    splits={"train": 1000, "val": 1000, "test": -1},
)
dataset_store = store(group="dataset")
dataset_store(qcml_data, name="qcml")
dataset_store(md17_benzene, name="md17_benzene")


loss_module = builds(
    LossModule,
    targets=["forces"],
    loss_types={"forces": "euclidean"},
    metrics={"forces": ["mae", "rmse", "euclidean", "huber", "mse"]},
    losses_per_mol=True,
)


@configure_main(extra_defaults=[{"dataset": "qcml"}])
def main(
    cfg: BaseConfig,  # noqa: ARG001
    model_run_dir: Path,
    loss_module: LossModule | None = loss_module,
    checkpoint_name: str = "best_model",
    dataset=MISSING,
    batch_size: int = 64,
    equivariance_metric: Literal["mse", "mae", "euclidean", "rmse", "huber"] = "euclidean",
    n_random_transforms: int = 128,
    n_samples: int = 50000,
    reflections: bool = False,
    rotations: bool = True,
    log_best_worst_conformers: int = 15,
    log_best_worst: int = 15,
    log_random: int = 15,
    log_uncertainty_performance_correlation: bool = True,
    ptdtype: Literal["float32", "float16", "bfloat16"] = "float32",
) -> None:
    mp.set_start_method("spawn", force=True)
    device, ctx = setup_device(), get_amp(ptdtype)
    config_path = model_run_dir / ".hydra" / "config.yaml"
    conf = load_from_yaml(config_path)
    model_conf = conf["train"]["model"]
    model = instantiate(model_conf)
    if loss_module is None:
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
    data = dataset(rank=0, copy_to_temp=False)
    # evaluation

    evaluation_metrics = compute_metrics(
        predictor,
        data,
        n_samples,
        device,
        ctx,
    )
    logger.info(f"Evaluation results:\n{pformat(evaluation_metrics)}")
    eval_results["metrics"] = evaluation_metrics
    equivariance_results = evaluate_equivariance(
        predictor,
        data,
        job_dir,
        device,
        ctx,
        metric=equivariance_metric,
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
    with results_path.open("w") as f:
        yaml.dump(eval_results, f)
    logger.info(f"Results saved to {results_path}")


@th.no_grad()
def compute_metrics(
    model: nn.Module,
    data: DatasetSplits,
    num_samples: int,
    device,
    ctx,
    batch_size: int = 1024,
) -> dict:
    prebatch_preprocessors = [center_positions_on_centroid]
    loader = DataLoader(
        data.splits[Split.test],
        batch_size=1024,
        collate_fn=partial(
            collate_fn,
            props=data.dataset_props,
            device=device,
            pre_batch_preprocessors=prebatch_preprocessors,
            post_batch_preprocessors=[],
        ),
        shuffle=False,
    )
    steps = num_samples // batch_size
    losses = []
    for batch in tqdm(loader, total=steps):
        if len(losses) >= steps:
            break
        with ctx:
            _, losses_batch = model(batch)
        losses.append(losses_batch)
    loss_keys = losses[0].keys()
    mean_losses = {k: th.cat([l[k] for l in losses], dim=0).mean().item() for k in loss_keys}
    logger.info(f"Mean losses:\n{pformat(mean_losses)}")
    return mean_losses


@th.no_grad()
def evaluate_equivariance(
    model: nn.Module,
    data: DatasetSplits,
    job_dir: Path,
    device,
    ctx,
    metric: Literal["mse", "mae", "euclidean", "rmse", "huber"],
    batch_size: int,
    n_random_transforms: int,
    n_samples: int,
    reflections: bool,
    rotations: bool,
    log_best_worst_conformers: int,
    log_best_worst: int,
    log_random: int,
    log_uncertainty_performance_correlation: bool,
) -> dict:
    assert batch_size % n_random_transforms == 0, "batch_size must be divisible by n_random_transforms"
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
        conformer_force_losses.append(losses[f"forces_{metric!s}"].view(-1, n_random_transforms))
    conformer_force_losses = th.cat(conformer_force_losses, dim=0)  # (n, n_random_transforms)
    conformer_data = [
        {k: v[i] for k, v in conf_data.items()} for conf_data in conformer_data for i in range(batch_size)
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
        ax.scatter(conformer_force_losses_mean.cpu(), equivariance_std.cpu(), alpha=0.2, s=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Conformer force loss ({metric!s})")
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
        export_dir.mkdir(exist_ok=True)
        for i, conf in enumerate(best_conformer_data):
            export_xyz(
                conf,
                export_dir,
                f"best_conformer_{i}",
            )
            plot_invariance(model, conf, ctx, export_dir, f"best_conformer_{i}")

        for i, conf in enumerate(worst_conformer_data):
            export_xyz(
                conf,
                export_dir,
                f"worst_conformer_{i}",
            )
            plot_invariance(model, conf, ctx, export_dir, f"worst_conformer_{i}")
    if log_best_worst > 0:
        sorted_args = conformer_force_losses_mean.argsort()
        best_conformers_idx = sorted_args[:log_best_worst]
        worst_conformers_idx = sorted_args[-log_best_worst:]
        best_conformer_data = [conformer_data[i] for i in best_conformers_idx]
        worst_conformer_data = [conformer_data[i] for i in worst_conformers_idx]

        ## save conformers
        export_dir = Path(job_dir) / "eval_results" / "best_worst"
        export_dir.mkdir(exist_ok=True)
        for i, conf in enumerate(best_conformer_data):
            export_xyz(
                conf,
                export_dir,
                f"best_conformer_{i}",
            )
            plot_invariance(model, conf, ctx, export_dir, f"best_conformer_{i}")
        for i, conf in enumerate(worst_conformer_data):
            export_xyz(
                conf,
                export_dir,
                f"worst_conformer_{i}",
            )
            plot_invariance(model, conf, ctx, export_dir, f"worst_conformer_{i}")
    if log_random > 0:
        random_idx = th.randint(0, len(conformer_data), (log_random,))
        random_conformer_data = [conformer_data[i] for i in random_idx]
        export_dir = Path(job_dir) / "eval_results" / "random"
        export_dir.mkdir(exist_ok=True)
        for i, conf in enumerate(random_conformer_data):
            export_xyz(
                conf,
                export_dir,
                f"random_conformer_{i}",
            )
            plot_invariance(model, conf, ctx, export_dir, f"random_conformer_{i}")

    return equivariance_results


def get_rotation_from_axis_angle(axis, angle) -> th.Tensor:
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
    identity = th.eye(3, device=axis.device).unsqueeze(0).expand(batch_dim, -1, -1)  # (n, 3, 3)
    rotation_matrices = cos_theta * identity + sin_theta * K + (1 - cos_theta) * outer_product  # (n, 3, 3)
    return rotation_matrices


@th.no_grad()
def plot_invariance(model, data, ctx, export_dir, export_name) -> Path:
    axes = th.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        device=data[Props.positions].device,
        dtype=th.float32,
    )
    n_interploations = 120
    angles = th.linspace(0, 2 * th.pi, n_interploations, device=data[Props.positions].device)  # (n_interploations,)
    axes = axes.repeat_interleave(n_interploations, dim=0)  # (3 * n_interploations, 3)
    angles = angles.repeat(3)  # (3 * n_interploations,)
    R = get_rotation_from_axis_angle(axes, angles)  # (3 * n_interploations, 3, 3)
    data = {k: v.unsqueeze(0).repeat(n_interploations * 3, *(len(v.shape) * [1])) for k, v in data.items()}
    data[Props.positions] = th.einsum("bij,bnj->bni", R, data[Props.positions])  # (3 * n_interploations, s, 3)
    data[Props.forces] = th.einsum("bij,bnj->bni", R, data[Props.forces])
    with ctx:
        _, losses = model(data)

    loss_x, loss_y, loss_z = losses[Props.forces].cpu().chunk(3, dim=0)  # (n_interploations,) * 3
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
    dataset_store.add_to_hydra_store()
    run(main)
