from pathlib import Path
from urllib import request

import numpy as np
from frozendict import frozendict
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from torch.utils.data import Subset

from .datasets import NPZDataset

md17_props = frozendict(
    {
        Props.energy: "E",
        Props.atomic_numbers: "z",
        Props.forces: "F",
        Props.positions: "R",
    }
)

_filenames = frozendict(
    {
        "aspirin": "md17_aspirin.npz",
        "azobenzene": "azobenzene_dft.npz",
        "benzene": "md17_benzene2017.npz",
        "ethanol": "md17_ethanol.npz",
        "malonaldehyde": "md17_malonaldehyde.npz",
        "naphthalene": "md17_naphthalene.npz",
        "paracetamol": "paracetamol_dft.npz",
        "salicylic_acid": "md17_salicylic.npz",
        "toluene": "md17_toluene.npz",
        "uracil": "md17_uracil.npz",
        "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
        "DHA": "md22_DHA.npz",
        "stachyose": "md22_stachyose.npz",
        "AT-AT": "md22_AT-AT.npz",
        "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
        "buckyball-catcher": "md22_buckyball-catcher.npz",
        "double-walled_nanotube": "md22_double-walled_nanotube.npz",
    }
)


def download_md17_22_dataset(dataset_path: Path, molecule: str) -> None:
    logger.info("Downloading MD17 dataset")
    # Download dataset to dataset_path
    mol_path = dataset_path / _filenames[molecule]
    url = "http://www.quantum-machine.org/gdml/data/npz/" + _filenames[molecule]
    request.urlretrieve(url, mol_path)  # noqa: S310
    logger.info(f"Downloaded {url} to {dataset_path}")


def get_md17_22_dataset(
    rank: int,
    data_dir: Path,
    molecule_name: str,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> DatasetSplits:
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}
    data_path = data_dir / "md17_22"
    data_path.mkdir(parents=True, exist_ok=True)
    if molecule_name not in _filenames:
        raise ValueError(f"Unknown molecule {molecule_name=}, expected one of {_filenames.keys()}")

    file_path = data_path / _filenames[molecule_name]

    if not file_path.exists() and rank == 0:
        logger.info(f"Md17 dataset not found, downloading to {data_path}")
        download_md17_22_dataset(data_path, molecule_name)

    dist.barrier()
    dataset = NPZDataset(file_path, md17_props)

    index_array = np.arange(len(dataset))
    train_val, test = train_test_split(
        index_array, test_size=splits["train"] + splits["val"], random_state=seed, stratify=dataset.file_indices
    )
    train, val = train_test_split(
        train_val, test_size=splits["val"], random_state=seed, stratify=dataset.file_indices[train_val]
    )

    datasets = {
        Split.train: Subset(dataset, train),
        Split.val: Subset(dataset, val),
        Split.test: Subset(dataset, test),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=md17_props,
    )
