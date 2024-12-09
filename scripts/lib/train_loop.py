from itertools import islice
from pathlib import Path
from pprint import pformat
from typing import Literal

import torch as th
from lib.ema import EMAModel
from lib.lr_scheduler import LRScheduler
from lib.types import Split
from lib.utils.checkpoint import save_checkpoint
from lib.utils.dist import get_amp
from lib.utils.log import log_dict
from lib.utils.wandb import WandBConfig
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Predictor(nn.Module):
    def __init__(self, encoder, loss_module) -> None:
        super().__init__()
        self.encoder = encoder
        self.loss_module = loss_module

    def forward(self, inputs) -> tuple[dict, dict]:
        out = self.encoder(inputs)
        loss = self.loss_module(out, inputs)
        return out, loss


def train_loop(
    rank: int,
    model: Predictor,
    loaders: dict[Split, DataLoader],
    optimizer: Optimizer,
    save_dir: Path,
    start_step: int,
    total_steps: int,
    grad_accum_steps: int,
    eval_samples: int,
    log_interval: int = -1,
    eval_interval: int = -1,
    save_interval: int = -1,
    lr_scheduler: LRScheduler = None,
    ema: EMAModel | None = None,
    wandb: WandBConfig | None = None,
    clip_grad: float = 1.0,
    ptdtype: Literal["float32", "bfloat16", "float16"] = "float32",
) -> None:
    lr_scheduler = LRScheduler() if lr_scheduler is None else lr_scheduler
    ctx = get_amp(ptdtype)
    save_dir.mkdir(exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    effective_total_train_steps = total_steps * grad_accum_steps

    train_loss = 0.0
    epoch = start_step // len(loaders[Split.train])
    best_val = None
    train_iter = iter(loaders[Split.train])

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
                loss_dict = {k: loss.item() for k, loss in losses.items()}
                loss_dict["lr"] = optimizer.param_groups[0]["lr"]
                log_dict(loss_dict, real_step, log_wandb=wandb)

                # eval
            if (real_step + 1) % eval_interval == 0 and rank == 0:
                del loss, out
                th.cuda.empty_cache()
                val_loss = evaluate(model, loaders[Split.val], ctx, ema, eval_samples)
                lr_scheduler.step_on_loss(real_step, val_loss)
                log_dict(
                    val_loss,
                    real_step - 1,
                    log_wandb=wandb,
                    key_suffix=f"_mae_{Split.val}",
                )

                if best_val is None or val_loss["total"] < best_val:
                    best_val = val_loss["total"]
                    test_loss = evaluate(model, loaders[Split.test], ctx, ema, eval_samples)
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
                        log_wandb=wandb,
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
    return model


def evaluate_loop(model, loader, ctx, ema, eval_samples) -> dict:
    total_losses = None
    steps = eval_samples // loader.batch_size
    if ema is not None:
        ema.apply_shadow()
    for batch in islice(loader, steps):
        with ctx:
            _, losses = model(batch)
        total_losses = {k: (v + total_losses[k] if total_losses else v) for k, v in losses.items()}
    if ema is not None:
        ema.restore()
    return total_losses


def eval_try_without_grad(model, loader, ctx, ema, eval_samples) -> dict:
    try:
        with th.no_grad():
            total_losses = evaluate_loop(model, loader, ctx, ema, eval_samples)
    except Exception:
        logger.warning("Error during inference with torch.no_grad, trying again with gradients")
        total_losses = evaluate_loop(model, loader, ctx, ema, eval_samples)
    return total_losses


def evaluate(model, loader, ctx, ema, eval_samples) -> dict:
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
