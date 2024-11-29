#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data /home/maxi/MOLECULAR_ML/5_refactored_repo/container.sif python
from functools import partial
from pathlib import Path
from pprint import pformat
from itertools import islice

from hydra_zen import builds
from hydra_zen.typing import Partial
from loguru import logger
import torch.multiprocessing as mp
import numpy as np
import torch as th
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from conf.base_conf import configure_main, BaseConfig

from lib.types import Property as DatasetSplits, PipelineConfig, Split
from lib.ema import EMAModel
from lib.lr_scheduler import get_lr_scheduler
from lib.loss import LossModule
from lib.models import PairEncoder, get_pair_encoder_pipeline_config
from lib.datasets import get_qcml_dataset
from lib.data.loaders import get_loaders
from lib.utils.checkpoint import load_checkpoint, save_checkpoint
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from lib.utils.dist import setup_dist, setup_device, cleanup_dist
from lib.utils.log import log_dict

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
    warmup_steps=2000,
    min_lr=1e-7,
)
loss_module = builds(
    LossModule,
    targets=["forces"],
    loss_types={"forces": "mae"},
)
pair_encoder_model = builds(
    PairEncoder,
    n_layers=3,
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
    target_heads=["forces"],
    head_project_down=True,
)
pair_encoder_data_config = builds(
    get_pair_encoder_pipeline_config,
    augmentation_mult=2,
    random_rotation=True,
    random_reflection=True,
    center_positions=True,
    dynamic_batch_size_cutoff=29,
)
qcml_data = pbuilds(
    get_qcml_dataset,
    data_dir="/home/maxi/MOLECULAR_ML/5_refactored_repo/data_ar",
    dataset_name="qcml_unified_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=True,
)


def train(
    rank: int,
    port: str,
    world_size: int,
    cfg: BaseConfig,
    model: nn.Module = pair_encoder_model,
    data: DatasetSplits = qcml_data,
    pipeline_conf: PipelineConfig = pair_encoder_data_config,
    loss: LossModule = loss_module,
    lr_scheduler: Partial[callable] | None = p_cosine_scheduler,
    ema: Partial[EMAModel] | None = p_ema,
    optimizer: Partial[th.optim.Optimizer] = p_optim,
    batch_size: int = 256,
    total_steps: int = 200_000,
    lr: float = 5e-4,
    grad_accum_steps: int = 1,
    log_interval: int = 5,
    eval_interval: int = 1000,
    save_interval: int = 50000,
    eval_samples: int = 50000,
    clip_grad: float = 1.0,
    checkpoint_path: str | None = None,
):
    setup_dist(rank, world_size, port=port)
    try:
        ctx, device = setup_device(rank)
        logger.info(f"Running on rank {rank} with device {device}")

        # model
        ddp_args = {
            "device_ids": ([rank] if cfg.runtime.device == "cuda" else None),
        }
        model = Predictor(model, loss).to(device)
        model = DDP(model, **ddp_args)
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

        train_loss = 0.0
        epoch = 0
        best_val = None
        train_iter = iter(loaders[Split.train])
        start_step = 0

        if checkpoint_path != None:
            start_step = load_checkpoint(model, checkpoint_path, optimizer, ema)
            epoch = start_step // len(loaders[Split.train])
            dist.barrier()

        lr_scheduler = lr_scheduler(
            optimizer, lr, lr_decay_steps=total_steps
        )  # init after checkpoint to load lr
        save_dir = cfg.runtime.out_dir / "ckpts"
        save_dir.mkdir(exist_ok=True)
        optimizer.zero_grad(set_to_none=True)
        effective_total_train_steps = total_steps * grad_accum_steps

        for step in range(effective_total_train_steps):
            if step < start_step * grad_accum_steps:
                continue
            with ctx:
                batch = next(train_iter, None)
                if batch is None:
                    train_iter = iter(loaders[Split.train])
                    epoch += 1
                    logger.info(f"Epoch {epoch}")
                    batch = next(train_iter)
                out, losses = model(batch)
                loss = losses["total"]
                loss /= grad_accum_steps
            loss.backward()
            train_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                th.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                if ema is not None:
                    ema.update()
                optimizer.zero_grad(set_to_none=True)
                real_step = step // grad_accum_steps
                lr_scheduler.step(real_step)

                # logging
                if real_step % log_interval == 0 and rank == 0:
                    loss_dict = {k: l.item() for k, l in losses.items()}
                    loss_dict["lr"] = optimizer.param_groups[0]["lr"]
                    log_dict(loss_dict, real_step, log_wandb=cfg.wandb)

                # eval
                if (real_step + 1) % eval_interval == 0 and rank == 0:
                    del loss, out
                    th.cuda.empty_cache()
                    val_loss = evaluate(
                        model, loaders[Split.val], ctx, ema, eval_samples
                    )
                    lr_scheduler.step_on_loss(real_step, val_loss)
                    log_dict(
                        val_loss,
                        real_step - 1,
                        log_wandb=cfg.wandb,
                        key_suffix=f"_mae_{Split.val}",
                    )

                    if best_val is None or val_loss["total"] < best_val:
                        best_val = val_loss["total"]
                        test_loss = evaluate(
                            model, loaders[Split.test], ctx, ema, eval_samples
                        )
                        save_checkpoint(
                            model.module.encoder,
                            optimizer,
                            real_step,
                            Path(save_dir) / "best_model.pth",
                            ema,
                        )
                        log_dict(
                            test_loss,
                            real_step - 1,
                            log_wandb=cfg.wandb,
                            key_suffix=f"_mae_{Split.test}",
                        )
                    train_loss = 0.0

                # checkpointing
                if (real_step + 1) % save_interval == 0 and rank == 0:
                    save_checkpoint(
                        model.module.encoder,
                        optimizer,
                        real_step,
                        save_dir / f"model_{real_step}.pth",
                        ema,
                    )

    finally:
        cleanup_dist()


p_train_func = pbuilds_full(train)


def evaluate_loop(model, loader, ctx, ema, eval_samples):
    total_losses = None
    steps = eval_samples // loader.batch_size
    if ema is not None:
        ema.apply_shadow()
    for batch in islice(loader, steps):
        with ctx:
            _, losses = model(batch)
        total_losses = {
            k: (v + total_losses[k] if total_losses else v) for k, v in losses.items()
        }
    if ema is not None:
        ema.restore()
    return total_losses


def eval_try_without_grad(model, loader, ctx, ema, eval_samples):
    try:
        with th.no_grad():
            total_losses = evaluate_loop(model, loader, ctx, ema, eval_samples)
    except Exception:
        logger.warning(
            f"Error during inference with torch.no_grad, trying again with gradients"
        )
        total_losses = evaluate_loop(model, loader, ctx, ema, eval_samples)
    return total_losses


def evaluate(model, loader, ctx, ema, eval_samples):
    model.eval()
    logger.info(f"Evaluating on {eval_samples} samples")
    steps = eval_samples // loader.batch_size
    total_losses = eval_try_without_grad(model, loader, ctx, None, eval_samples)
    losses = {f"{k}": v.item() / steps for k, v in total_losses.items()}
    if ema is not None:
        ema_losses = eval_try_without_grad(model, loader, ctx, ema, eval_samples)
        losses.update({f"ema_{k}": v.item() / steps for k, v in ema_losses.items()})
    logger.info(f"Losses: {pformat(losses)}")
    model.train()
    return losses


class Predictor(nn.Module):
    def __init__(self, encoder, loss_module):
        super().__init__()
        self.encoder = encoder
        self.loss_module = loss_module

    def forward(self, inputs):
        out = self.encoder(inputs)
        loss = self.loss_module(out, inputs)
        return out, loss


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
