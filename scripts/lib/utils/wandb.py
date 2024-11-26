import os
from dataclasses import dataclass, fields
from typing import Self

from dotenv import load_dotenv
import wandb
from wandb.wandb_run import Run

from loguru import logger


@dataclass
class WandBConfig:
    """Configures WandB from environment variables."""

    WANDB_API_KEY: str
    WANDB_ENTITY: str
    WANDB_PROJECT: str

    @classmethod
    def from_env(cls) -> Self | None:
        """Read WandB environment variables.

        Returns
        -------
        Populated `WandBConfig` or None if environment variables could not be found.
        """
        config = None
        load_dotenv()

        try:
            config = cls(
                **{field.name: os.environ[field.name] for field in fields(cls)}
            )
        except KeyError:
            logger.info(
                "Could not load WandB config from environment variables or .env file."
            )

        return config


class WandBRun:
    """Initializes a WandB run from environment variables."""

    def __init__(
        self,
        entity: str | None = None,
        project: str | None = None,
        **kwargs,
    ) -> None:
        if (config := WandBConfig.from_env()) is not None:
            entity = config.WANDB_ENTITY
            project = config.WANDB_PROJECT

        run = wandb.init(entity=entity, project=project, **kwargs)

        if not isinstance(run, Run):
            raise TypeError("Could not initalize WandB run.")

        self.run = run
