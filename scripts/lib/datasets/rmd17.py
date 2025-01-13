import tarfile
from pathlib import Path
from urllib import request

import numpy as np
from frozendict import frozendict
from lib.datasets.datasets import NPZDataset
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from torch.utils.data import Subset

rmd17_props = frozendict(
    {
        Props.energy: "energies",
        Props.atomic_numbers: "nuclear_charges",
        Props.forces: "forces",
        Props.positions: "coords",
    }
)


_filenames = frozendict(
    {
        "aspirin": "rmd17_aspirin.npz",
        "azobenzene": "rmd17_azobenzene.npz",
        "benzene": "rmd17_benzene.npz",
        "ethanol": "rmd17_ethanol.npz",
        "malonaldehyde": "rmd17_malonaldehyde.npz",
        "naphthalene": "rmd17_naphthalene.npz",
        "paracetamol": "rmd17_paracetamol.npz",
        "salicylic_acid": "rmd17_salicylic.npz",
        "toluene": "rmd17_toluene.npz",
        "uracil": "rmd17_uracil.npz",
    }
)


def download_rmd17_dataset(dataset_path: Path) -> None:
    logger.info("Downloading RMD17 dataset")
    # Download dataset to dataset_path
    tar_path = dataset_path / "md17.tar.gz"

    url = "https://figshare.com/ndownloader/files/23950376"
    request.urlretrieve(url, tar_path)  # noqa: S310
    logger.info(f"Downloaded {url} to {dataset_path}")

    logger.info("Extracting data...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(dataset_path, filter="data")

    tar_path.unlink()
    logger.info("Extracted data & deleted raw data")


def get_rmd17_dataset(
    rank: int,
    data_dir: Path,
    molecule_name: str,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> DatasetSplits:
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}
    data_path = data_dir / "rmd17"
    data_path.mkdir(parents=True, exist_ok=True)
    if molecule_name not in _filenames:
        raise ValueError(f"Unknown molecule {molecule_name=}, expected one of {_filenames.keys()}")

    file_path = data_path / "npz_data" / _filenames[molecule_name]

    if not file_path.exists() and rank == 0:
        logger.info(f"Md17 dataset not found, downloading to {data_path}")
        download_rmd17_dataset(data_dir)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    dataset = NPZDataset(file_path, props=rmd17_props, force_unit="kcal/(mol·Å)")

    print(dataset[0])

    index_array = np.arange(len(dataset))
    train_val, test = train_test_split(
        index_array, test_size=splits["train"] + splits["val"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=splits["val"], random_state=seed
    )

    datasets = {
        Split.train: Subset(dataset, train),
        Split.val: Subset(dataset, val),
        Split.test: Subset(dataset, test),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=rmd17_props,
    )

get_rmd17_dataset(0, Path("."), "aspirin")