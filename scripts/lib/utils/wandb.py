import os
from dataclasses import dataclass, fields
from typing import Self

from dotenv import load_dotenv
from loguru import logger

import wandb
from wandb.wandb_run import Run


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
            config = cls(**{field.name: os.environ[field.name] for field in fields(cls)})
        except KeyError:
            logger.info("Could not load WandB config from environment variables or .env file.")

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

        run = wandb.init(entity=entity, project=project, **kwargs, config={})

        if not isinstance(run, Run):
            raise TypeError("Could not initalize WandB run.")
        self.run_id = run.id
        self.run = run
        self.project = project
        self.entity = entity
        self.config = dict(run.config)

    def reinit(self) -> None:
        """Recreate the run, for instance on a subprocess"""
        self.run = wandb.init(
            entity=self.entity, project=self.project, id=self.run_id, resume="must", config=self.config
        )

    def set_config(self, config: dict) -> None:
        """Update the run's config."""
        self.run.config.update(config)
        self.config = dict(self.run.config)
