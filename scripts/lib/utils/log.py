import numpy as np
from loguru import logger

import wandb


def log_str(d: dict, step: int) -> str:
    return f"Step {step}: " + " | ".join([f"{k}: {v:.3e}" for k, v in d.items() if isinstance(v, (int, float))])


def log_dict(d, step, log_wandb=False, key_suffix="") -> None:
    log_dict = {f"{k}{key_suffix}": v for k, v in d.items()}
    logger.info(log_str(log_dict, step))
    if log_wandb:
        # seperate all metrics that are scalar into a dict
        scalar_dict = {k: v for k, v in log_dict.items() if isinstance(v, (int, float))}
        wandb.log(scalar_dict, step=step)

        # plot all multidimensional metrics as histograms
        multidim_keys = set(log_dict.keys()) - set(scalar_dict.keys())
        for k in multidim_keys:
            v = log_dict[k]
            wandb.log({k: wandb.Histogram(v)}, step=step)
