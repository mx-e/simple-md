#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data /home/korjakow/simple-md/container.sif python
from functools import partial
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import builds, instantiate, load_from_yaml
from hydra_zen.typing import Partial
from lib.data.loaders import get_loaders
from lib.datasets import get_qcml_dataset
from lib.ema import EMAModel
from lib.loss import LossModule
from lib.lr_scheduler import LRScheduler, get_lr_scheduler
from lib.models import PairEncoder, get_pair_encoder_pipeline_config
from lib.models.pair_encoder import NodeLevelRegressionHead
from lib.train_loop import Predictor, train_loop
from lib.types import PipelineConfig
from lib.types import Property as DatasetSplits
from lib.types import Property as Props
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
loss_module_dipole = builds(
    LossModule,
    targets=["dipole"],
    loss_types={"dipole": "euclidean"},
    metrics={"dipole": ["mae", "mse", "euclidean"]},
    compute_metrics_train=False,
)
loss_module_forces = builds(
    LossModule,
    targets=["forces"],
    loss_types={"forces": "mse"},
    metrics={"forces": ["mae", "mse", "euclidean"]},
    compute_metrics_train=False,
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
    data_dir="data/data_arrecord",
    dataset_name="qcml_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=True,
)
ft_loop = pbuilds(
    train_loop,
    log_interval=5,
    eval_interval=5000,
    save_interval=50000,
    eval_samples=50000,
    clip_grad=1.0,
    ptdtype="float32",
)


def finetune(
    rank: int,
    port: str,
    world_size: int,
    cfg: BaseConfig,
    pretrain_model_dir: Path,
    checkpoint_name: str = "best_model",
    data: DatasetSplits = qcml_data,
    optimizer: Partial[th.optim.Optimizer] = p_optim,
    train_loop: Partial[callable] = ft_loop,
    batch_size: int = 256,
    total_steps: int = 220_000,
    lr: float = 1e-4,
    grad_accum_steps: int = 1,
    lr_scheduler: Partial[callable] | None = p_cosine_scheduler,  # None = No schedule
    loss: LossModule | None = loss_module_dipole,  # None = Same as pretrain
    pipeline_conf: PipelineConfig | None = pair_encoder_data_config,  # None = Same as pretrain
    ema: Partial[EMAModel] | None = None,  # None = No EMA
) -> None:
    setup_dist(rank, world_size, port=port)
    try:
        device = setup_device(rank)

        # get model + loss + optim
        config_path = pretrain_model_dir / ".hydra" / "config.yaml"
        conf = load_from_yaml(config_path)
        model_conf = conf["train"]["model"]
        model = instantiate(model_conf)
        if loss is None:
            loss = instantiate(conf["train"]["loss"])
        model = Predictor(model, loss).to(device)
        ddp_args = {
            "device_ids": ([rank] if cfg.runtime.device == "cuda" else None),
        }
        model = DDP(model, **ddp_args)
        checkpoint_path = pretrain_model_dir / "ckpts" / f"{checkpoint_name}.pth"
        load_checkpoint(model.module.encoder, checkpoint_path)
        dist.barrier()
        if ema is not None and rank == 0:
            ema = ema(model.module, device=device)
            logger.info(f"Using EMA with decay {ema.decay}")
        else:
            ema = None
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params}")
        optimizer = optimizer(model.parameters(), lr=lr)
        lr_scheduler = (
            lr_scheduler(optimizer, lr, lr_decay_steps=total_steps)
            if lr_scheduler is not None
            else LRScheduler(optimizer, lr)
        )

        # handle cross-objective finetuning
        loss_pretrain = instantiate(conf["train"]["loss"])
        if loss_pretrain.targets != loss.targets:
            logger.info(
                f"Cross-objective finetuning detected: pretrain loss targets {loss_pretrain.targets} vs finetune loss targets {loss.targets}"
            )
            if not isinstance(model.module.encoder, PairEncoder):
                raise ValueError("Cross-objective finetuning only supported for PairEncoder models")
            logger.info(f"Modifying model to output {loss.targets} instead of {loss_pretrain.targets}")
            current_heads = model.module.encoder.heads
            new_heads = {target: head for target, head in current_heads.items() if target in loss.targets}
            new_head_targets = [target for target in loss.targets if target not in loss_pretrain.targets]
            for target in new_head_targets:
                new_heads[str(target)] = NodeLevelRegressionHead(
                    target=target,
                    embd_dim=model_conf.embd_dim,
                    cls_token=model_conf.cls_token,
                    activation=model_conf.activation,
                    head_dropout=model_conf.head_dropout,
                    project_down=model_conf.head_project_down,
                ).to(device)
            model.module.encoder.heads = nn.ModuleDict(new_heads)
            for target in new_head_targets:  # dataloading pipeline needs to get new targets
                pipeline_conf.needed_props.append(target)
            model = DDP(model.module, **ddp_args)  # rewrap model after modification

        # data + loaders
        data = data(rank)
        if pipeline_conf is None:
            try:
                pipeline_conf = instantiate(conf["train"]["pipeline_conf"])
                logger.info("Loaded pipeline config from pretraining config")
            except KeyError as e:
                raise ValueError(
                    "No pipeline config found in pretrain config - this might be an old checkpoint, please specify manually"
                ) from e
        loaders = get_loaders(
            rank=rank,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            world_size=world_size,
            device=device,
            dataset_splits=data,
            pipeline_config=pipeline_conf,
        )

        final_model = train_loop(
            rank=rank,
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            save_dir=cfg.runtime.out_dir / "ckpts",
            start_step=0,
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
            cfg.runtime.out_dir / "ckpts" / "model_final.pth",
            ema,
        )
    finally:
        cleanup_dist()


p_ft_func = pbuilds_full(finetune)


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,  # you must keep this argument
    pretrain_model_dir: str,
    ft: Partial[callable] = p_ft_func,
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    world_size = cfg.runtime.n_gpu if th.cuda.is_available() else 1
    logger.info(f"Running {world_size} process(es)")
    random_port = str(np.random.randint(20000, 50000))
    cfg.runtime.out_dir = get_hydra_output_dir()

    if world_size > 1:
        th.multiprocessing.spawn(
            ft,
            args=(random_port, world_size, cfg),
            nprocs=world_size,
            join=True,
            pretrain_model_dir=Path(pretrain_model_dir),
        )
    else:
        ft(rank=0, port=random_port, world_size=1, cfg=cfg, pretrain_model_dir=Path(pretrain_model_dir))


if __name__ == "__main__":
    run(main)
