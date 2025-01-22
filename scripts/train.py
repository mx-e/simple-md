#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif python
from functools import partial

import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import builds
from hydra_zen.typing import Partial
from lib.data.loaders import get_loaders
from lib.datasets import get_qcml_dataset
from lib.ema import EMAModel
from lib.loss import LossModule
from lib.lr_scheduler import get_lr_scheduler
from lib.models import PairEncoder, get_pair_encoder_pipeline_config
from lib.train_loop import Predictor, train_loop
from lib.types import PipelineConfig
from lib.types import Property as DatasetSplits
from lib.utils.checkpoint import load_checkpoint, save_checkpoint
from lib.utils.dist import cleanup_dist, setup_device, setup_dist
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

pbuilds = partial(builds, zen_partial=True)
pbuilds_full = partial(builds, zen_partial=True, populate_full_signature=True)

p_optim = pbuilds(
    th.optim.AdamW,
    weight_decay=1e-7,
)
p_ema = pbuilds(
    EMAModel,
    decay=0.9997,
)
p_no_scheduler = pbuilds(get_lr_scheduler)
p_cosine_scheduler = pbuilds(
    get_lr_scheduler,
    scheduler_type="cosine_warmup",
    warmup_steps=5000,
    min_lr=1e-7,
)
loss_module_forces = builds(
    LossModule,
    targets=["forces"],
    loss_types={"forces": "mse"},
    metrics={"forces": ["mae", "mse", "euclidean"]},
    compute_metrics_train=False,
)
loss_module_dipole = builds(
    LossModule,
    targets=["dipole"],
    loss_types={"dipole": "mae"},
    metrics={"dipole": ["mae", "mse", "euclidean"]},
    compute_metrics_train=False,
)

pair_encoder_model = builds(
    PairEncoder,
    n_layers=12,
    embd_dim=192,
    num_3d_kernels=128,
    cls_token=False,
    num_heads=12,
    activation="gelu",
    ffn_multiplier=2,
    attention_dropout=0.0,
    ffn_dropout=0.0,
    head_dropout=0.0,
    norm_first=True,
    norm="layer",
    decomposer_type="pooling",
    target_heads=["forces"],
    head_project_down=True,
    compose_dipole_from_charges=False,
)
pair_encoder_data_config = builds(
    get_pair_encoder_pipeline_config,
    augmentation_mult=2,
    random_rotation=True,
    random_reflection=True,
    center_positions=True,
    dynamic_batch_size_cutoff=29,
    include_dipole=True,
)
qcml_data = pbuilds(
    get_qcml_dataset,
    data_dir="/data/data_arrecord",
    dataset_name="qcml_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=True,
)
pretrain_loop = pbuilds(
    train_loop,
    log_interval=5,
    eval_interval=5000,
    save_interval=50000,
    eval_samples=50000,
    clip_grad=1.0,
    ptdtype="float32",
)


def train(
    rank: int,
    port: str,
    world_size: int,
    cfg: BaseConfig,
    model: nn.Module = pair_encoder_model,
    data: DatasetSplits = qcml_data,
    pipeline_conf: PipelineConfig = pair_encoder_data_config,
    loss: LossModule = loss_module_forces,
    train_loop: Partial[callable] | None = pretrain_loop,
    lr_scheduler: Partial[callable] | None = p_cosine_scheduler,
    ema: Partial[EMAModel] | None = p_ema,
    optimizer: Partial[th.optim.Optimizer] = p_optim,
    batch_size: int = 256,
    total_steps: int = 220_000,
    lr: float = 5e-4,
    grad_accum_steps: int = 1,
    checkpoint_path: str | None = None,
) -> None:
    setup_dist(rank, world_size, port=port)
    try:
        device = setup_device(rank)
        if world_size > 1 and rank == 0:
            cfg.wandb.reinit()  # move wandb session to spawned process for multi-gpu
        # model
        ddp_args = {
            "device_ids": ([rank] if cfg.runtime.device == "cuda" else None),
        }
        predictor = Predictor(model, loss).to(device)
        model = DDP(predictor, **ddp_args)
        if ema is not None and rank == 0:
            ema = ema(model.module, device=device)
            logger.info(f"Using EMA with decay {ema.decay}")
        else:
            ema = None
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params}")
        optimizer = optimizer(model.parameters(), lr=lr)

        # data
        data = data(rank)
        loaders = get_loaders(
            rank=rank,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            world_size=world_size,
            device=device,
            dataset_splits=data,
            pipeline_config=pipeline_conf,
        )
        start_step = 0
        if checkpoint_path is not None:
            start_step = load_checkpoint(predictor.encoder, checkpoint_path, optimizer, ema)
            model = DDP(predictor, **ddp_args)
            dist.barrier()

        lr_scheduler = lr_scheduler(optimizer, lr, lr_decay_steps=total_steps)  # init after checkpoint to load lr
        save_dir = cfg.runtime.out_dir / "ckpts"

        final_model = train_loop(
            rank=rank,
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            save_dir=save_dir,
            start_step=start_step,
            total_steps=total_steps,
            grad_accum_steps=grad_accum_steps,
            lr_scheduler=lr_scheduler,
            ema=ema,
            wandb=cfg.wandb,
        )

        save_checkpoint(
            final_model.module.encoder,
            optimizer,
            total_steps,
            save_dir / "model_final.pth",
            ema,
        )
    finally:
        cleanup_dist()


p_train_func = pbuilds_full(train)


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,  # you must keep this argument
    cfg_version: str = "1.1",  # noqa: ARG001 saving config version to track changes in signatures eg for finetuning
    train: Partial[callable] = p_train_func,
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    world_size = cfg.runtime.n_gpu if th.cuda.is_available() else 1
    logger.info(f"Running {world_size} process(es)")
    rng = np.random.RandomState()  # port selection should be truly random
    random_port = str(
        rng.randint(20000, 50000),
    )
    cfg.runtime.out_dir = get_hydra_output_dir()

    if world_size > 1:
        th.multiprocessing.spawn(
            train,
            args=(random_port, world_size, cfg),
            nprocs=world_size,
            join=True,
        )
    else:
        train(rank=0, port=random_port, world_size=1, cfg=cfg)


if __name__ == "__main__":
    run(main)
