from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
import socket

from hydra_zen import builds, store
import torch as th

from lib.utils.wandb import WandBRun
from lib.utils.job import Job, SweepJob, SlurmConfig
from lib.utils.helpers import get_hydra_output_dir


@dataclass
class RuntimeInfo:
    device: str = field(default_factory=lambda: "cuda" if th.cuda.is_available() else "cpu")
    out_dir: Path | None = None
    n_gpu: int = field(default_factory=th.cuda.device_count)
    node_hostname: str = field(default_factory=socket.gethostname)


@dataclass
class BaseConfig:
    seed: int = 42
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)
    loglevel: Literal["debug", "info"] = "debug"
    wandb: WandBRun | None = None
    job: Job | None = None


BaseSlurmConfig = builds(
    SlurmConfig,
    partition="gpu-2d",
    cpus_per_task=8,
    gpus_per_task=1,
    memory_gb=32,
    nodes=1,
    tasks_per_node=1,
    exclude="head001",
    constraint="h100|80gb",
)

# get main script path
sif_path = Path(__file__).resolve().parent.parent.parent / "container.sif"

BaseJobConfig = builds(
    Job,
    image=sif_path,
    kwargs={},
    slurm_config=BaseSlurmConfig,
)

BaseSweepConfig = builds(
    SweepJob,
    num_workers=6,
    sweep_id="new_md17_exp",
    # parameters={"pretrain_model_dir":["/data/models/pretrained/forces_4gpuday", "/data/models/pretrained/forces_16gpuday"],"cfg.seed": [1,2,3]},
    # parameters={"ft.dataset": ["rmd17_aspirin","md17_aspirin","md17_ethanol","md17_malonaldehyde","md17_naphthalene","md17_salicylic_acid","md17_toluene","md17_uracil","md17_azobenzene","md17_benzene","md17_paracetamol","md22_Ac_Ala3_NHMe","md22_DHA","md22_stachyose","md22_AT_AT","md22_AT_AT_CG_CG","md22_buckyball_catcher","md22_double_walled_nanotube","rmd17_azobenzene","rmd17_benzene","rmd17_ethanol","rmd17_malonaldehyde","rmd17_naphthalene","rmd17_paracetamol","rmd17_salicylic_acid","rmd17_toluene","rmd17_uracil","qm7x_pbe0","qm7x","ko2020_ag_cluster","ko2020_AuMgO","ko2020_Carbon_chain","ko2020_NaCl"]},
    # parameters={"pretrain_model_dir":["/data/models/pretrained/forces_4gpuday", "/data/models/pretrained/forces_16gpuday"],"ft.dataset": ["md22_Ac_Ala3_NHMe","md22_DHA","md22_stachyose","md22_AT_AT","md22_AT_AT_CG_CG","md22_buckyball_catcher"]},
    parameters={
        "pretrain_model_dir": ["/data/models/pretrained/forces_4gpuday", "/data/models/pretrained/forces_16gpuday"],
        "cfg.seed": [1, 2, 3],
        "ft.dataset": ["md17_aspirin", "md17_ethanol", "md17_naphthalene", "md17_salicylic_acid"],
    },
    # parameters={"pretrain_model_dir":["/data/models/pretrained/forces_4gpuday", "/data/models/pretrained/forces_16gpuday"],"cfg.seed": [1,2,3],"ft.dataset": ["rmd17_aspirin","rmd17_ethanol","rmd17_naphthalene","rmd17_salicylic_acid"]},
    # parameters={"pretrain_model_dir":["/data/models/pretrained/forces_4gpuday", "/data/models/pretrained/forces_16gpuday"],"ft.dataset": ["ko2020_ag_cluster","ko2020_AuMgO","ko2020_Carbon_chain","ko2020_NaCl"]},
    # parameters={"pretrain_model_dir":["/data/models/pretrained/forces_4gpuday", "/data/models/pretrained/forces_16gpuday"],"cfg.seed": [1,2,3],"ft.dataset": ["qm7x_pbe0","qm7x"]},
    builds_bases=(BaseJobConfig,),
)

BaseWandBConfig = builds(WandBRun, group=None, mode="online")


def configure_main(extra_defaults: list[dict] | None = None):
    wandb_config_store = store(group="cfg/wandb")
    wandb_config_store(BaseWandBConfig, name="log")

    job_config_store = store(group="cfg/job")
    job_config_store(BaseJobConfig, name="run")
    job_config_store(BaseSweepConfig, name="sweep")

    run_config = builds(BaseConfig, populate_full_signature=True)
    base_defaults = ["_self_", {"cfg/wandb": None}, {"cfg/job": None}]
    if extra_defaults is not None:
        base_defaults.extend(extra_defaults)

    def decorator(main_func):
        main_func_store = store(
            main_func,
            name="root",
            cfg=run_config,
            hydra_defaults=base_defaults,
        )

        return main_func_store

    return decorator
