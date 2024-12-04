from loguru import logger

import wandb


def log_str(d: dict, step: int) -> str:
    return f"Step {step}: " + " | ".join([f"{k}: {v:.3e}" for k, v in d.items()])


def log_dict(d, step, log_wandb=False, key_suffix="") -> None:
    log_dict = {f"{k}{key_suffix}": v for k, v in d.items()}
    logger.info(log_str(log_dict, step))
    if log_wandb:
        wandb.log(log_dict, step=step)
