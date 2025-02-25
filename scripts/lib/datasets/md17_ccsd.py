from pathlib import Path
from urllib import request

import numpy as np
from frozendict import frozendict
from lib.datasets.datasets import NPZDataset
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from loguru import logger
from sklearn.model_selection import train_test_split
from scripts.lib.datasets.utils import get_split_by_molecule_name
from torch import distributed as dist
from torch.utils.data import Subset

ethanol_ccsd_props = frozendict(
    {
        Props.energy: "E",
        Props.atomic_numbers: "z",
        Props.forces: "F",
        Props.positions: "R",
    }
)

paths = frozendict(
    {
        "ethanol": "ethanol_ccsd",
        "aspirin": "aspirin_ccsd",
        "toluene": "toluene_ccsd",
        "benzene": "benzene_ccsd",
        "malonaldehyde": "malonaldehyde_ccsd",
    }
)

file_names_train = frozendict(
    {
        "ethanol": "ethanol_ccsd_t-train.npz",
        "aspirin": "aspirin_ccsd-train.npz",
        "toluene": "toluene_ccsd_t-train.npz",
        "benzene": "benzene_ccsd_t-train.npz",
        "malonaldehyde": "malonaldehyde_ccsd_t-train.npz",
    }
)

file_names_test = frozendict(
    {
        "ethanol": "ethanol_ccsd_t-test.npz",
        "aspirin": "aspirin_ccsd-test.npz",
        "toluene": "toluene_ccsd_t-test.npz",
        "benzene": "benzene_ccsd_t-test.npz",
        "malonaldehyde": "malonaldehyde_ccsd_t-test.npz",
    }
)


def get_md17_ccsd_dataset(
    rank: int,
    data_dir: Path,
    molecule_name: str,
    seed: int = 42,
    copy_to_temp: bool = False,  # noqa: ARG001
    splits: dict[str, float] | None = None,
) -> DatasetSplits:
    data_path = data_dir / paths[molecule_name]
    data_path.mkdir(parents=True, exist_ok=True)
    data_path_train = data_path / file_names_train[molecule_name]
    data_path_test = data_path / file_names_test[molecule_name]

    assert data_path_train.exists(), f"Training data not found at {data_path_train}"
    assert data_path_test.exists(), f"Test data not found at {data_path_test}"

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    dataset_train = NPZDataset(data_path_train, ethanol_ccsd_props, force_unit="kcal/(mol·Å)", coord_unit="Å")
    dataset_test = NPZDataset(data_path_test, ethanol_ccsd_props, force_unit="kcal/(mol·Å)", coord_unit="Å")

    datasets = {Split.train: dataset_train, Split.val: dataset_test, Split.test: dataset_test}

    return DatasetSplits(
        splits=datasets,
        dataset_props=ethanol_ccsd_props,
    )
