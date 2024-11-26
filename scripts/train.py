#! /usr/bin/env -S apptainer exec /home/maxi/MOLECULAR_ML/5_refactored_repo/container.sif python
from functools import partial

from loguru import logger

from hydra_zen import builds, store
from hydra_zen.typing import Partial
import torch.multiprocessing as mp
import numpy as np

import torch as th
from torch import nn

from conf.base_conf import configure_main, BaseConfig
from lib.utils.run import run
from lib.utils.dist import setup_dist, setup_device
from lib.ema import EMAModel
from lib.lr_scheduler import get_lr_scheduler
from lib.loss import LossModule, LossType
from lib.types import Property as Props, LoadedDataset
from lib.models import PairEncoder
from lib.datasets import get_qcml_dataset

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
    warmup_steps=15000,
    min_lr=1e-7,
)
loss_module = builds(
    LossModule,
    targets=[Props.forces],
    loss_types={Props.forces: LossType.mae},
)
pair_encoder_model = builds(
    PairEncoder,
    n_layers=12,
    embd_dim=192,
    num_3d_kernels=128,
    cls_token=False,
    num_heads=12,
    activation="gelu",
    ffn_multiplier=4,
    attention_dropout=0.0,
    ffn_dropout=0.0,
    head_dropout=0.0,
    norm_first=True,
    norm="layer",
    decomposer_type="pooling",
    target_heads=[Props.forces],
    head_project_down=True,
)
qcml_data = builds(
    get_qcml_dataset,
    data_dir="/home/maxi/MOLECULAR_ML/5_refactored_repo/data_ar",
    dataset_name="qcml_unified_fixed_split_by_smiles",
    dataset_version="1.0.0",
    splits={"train": "train", "valid": "valid", "test": "test"},
    copy_to_temp=True,
)


def train(
    rank: int,
    port: str,
    world_size: int,
    cfg: BaseConfig,
    model: nn.Module = pair_encoder_model,
    data: LoadedDataset = qcml_data,
    loss: LossModule = loss_module,
    lr_scheduler: Partial[callable] | None = p_cosine_scheduler,
    ema: Partial[EMAModel] | None = p_ema,
    optimizer: Partial[th.optim.Optimizer] = p_optim,
    total_steps: int = 880_000,
    lr: float = 5e-4,
    grad_accum_steps: int = 1,
    log_interval: int = 5,
    eval_interval: int = 5000,
    eval_samples: int = 50000,
    clip_grad: float = 1.0,
):
    setup_dist(rank, world_size, port=port)
    ctx, device = setup_device(rank)


p_train_func = pbuilds_full(train)


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,  # you must keep this argument
    train: Partial[callable] = p_train_func,
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    world_size = cfg.runtime.n_gpu if th.cuda.is_available() else 1
    logger.info(f"Running {world_size} process(es)")
    random_port = str(np.random.randint(20000, 50000))
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
