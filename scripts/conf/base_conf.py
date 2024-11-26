from pathlib import Path
from dataclasses import dataclass

from hydra_zen import builds, store

from lib.utils.wandb import WandBRun
from lib.utils.job import Job, SweepJob, SlurmConfig


@dataclass
class BaseConfig:
    seed: int = 42
    wandb: WandBRun | None = None
    job: Job | None = None


BaseSlurmConfig = builds(
    SlurmConfig,
    cpus_per_task=3,
    gpus_per_task=0,
    memory_gb=16,
    nodes=1,
    tasks_per_node=1,
)

# get main script path
sif_path = Path(__file__).resolve().parent.parent.parent / "container.sif"

BaseJobConfig = builds(
    Job,
    partition="cpu-2h",
    image=sif_path,
    cluster="slurm",
    kwargs={},
    slurm_config=BaseSlurmConfig,
)

BaseSweepConfig = builds(
    SweepJob,
    num_workers=2,
    parameters={"cfg.seed": [42, 1337]},
    builds_bases=(BaseJobConfig,),
)

BaseWandBConfig = builds(WandBRun, group=None, mode="online")


def configure_main(main_func):
    wandb_config_store = store(group="cfg/wandb")
    wandb_config_store(BaseWandBConfig, name="log")

    job_config_store = store(group="cfg/job")
    job_config_store(BaseJobConfig, name="run")
    job_config_store(BaseSweepConfig, name="sweep")

    run_config = builds(BaseConfig, populate_full_signature=True)

    main_func_store = store(
        main_func,
        name="root",
        cfg=run_config,
        hydra_defaults=["_self_", {"cfg/wandb": None}, {"cfg/job": None}],
    )

    return main_func_store
