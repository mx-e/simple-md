#! /usr/bin/env -S apptainer exec /home/maxi/TEMP/ml-project-template/container.sif python

from loguru import logger

from conf.base_conf import configure_main, BaseConfig
from lib.utils.run import run


@configure_main
def train(
    cfg: BaseConfig,  # you must keep this argument
    bar: int = 42,
    foo: str = "hello",
    jup: bool = False,
    test: float = 2.2,
) -> None:
    logger.info("Running main function.")
    logger.info(f"Config: bar={bar}, foo={foo}, jup={jup}")
    logger.info(f"BaseConfig: {cfg}")


if __name__ == "__main__":
    run(train)
