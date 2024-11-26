from collections.abc import Callable
from functools import partial

from hydra_zen import MISSING, instantiate, store, zen
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb

from lib.utils.helpers import get_hydra_output_dir, seed_everything
from lib.utils.wandb import WandBRun


def pre_call(root_config: DictConfig, log_debug: bool = False) -> None:
    """Logs the config, sets the seed and initializes a WandB run before config instantiation.

    Args:
    ----
        root_config: Unresolved config.
        log_debug: Whether to log the config, seed and output path.
    """
    assert (
        root_config is not None and root_config["cfg"] is not None
    ), "Config must contain 'conf' at root-level."
    config: DictConfig = root_config["cfg"]
    assert (
        "seed" in config and "job" in config and "wandb" in config
    ), "Do not edit the BaseConfig schema without updating the pre_call function."
    seed = config.get("seed", MISSING)
    job = config.get("job", MISSING)
    if job is not MISSING:
        return

    if seed is not MISSING:
        seed_everything(seed)
        logger.debug(f"Set seed to {seed}.")
    else:
        logger.warn("No seed was configured! Run may not be reproducible.")

    output_path = get_hydra_output_dir()
    logger.debug(f"Saving outputs in {output_path}")

    if (wandb_config := config.get("wandb")) is not None:
        wandb_run: WandBRun = instantiate(wandb_config)
        wandb_run.run.config.update(OmegaConf.to_container(root_config))
        wandb.save(output_path / ".hydra/*", base_path=output_path, policy="now")


def run(main_function: Callable, log_debug=True) -> None:
    """Configure and run a given function using hydra-zen.

    Args:
    ----
        main_function: Function to configure and run.
    """
    store.add_to_hydra_store()
    zen(
        main_function,
        pre_call=partial(pre_call, log_debug=log_debug),
        resolve_pre_call=False,
    ).hydra_main(
        config_name="root",
        config_path=None,
        version_base="1.3",
    )
