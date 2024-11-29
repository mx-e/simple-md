from loguru import logger
import wandb

log_str = lambda dict, step: f"Step {step}: " + " | ".join(
    [f"{k}: {v:.3e}" for k, v in dict.items()]
)


def log_dict(dict, step, log_wandb=False, key_suffix=""):
    log_dict = {f"{k}{key_suffix}": v for k, v in dict.items()}
    logger.info(log_str(log_dict, step))
    if log_wandb:
        wandb.log(log_dict, step=step)
