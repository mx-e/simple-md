from pathlib import Path

import tensorflow_datasets as tfds
import torch.distributed as dist
from frozendict import frozendict
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from loguru import logger

qm7x_pbe0_props = frozendict(
    {
        Props.energy: "pbe0_energy",
        Props.formation_energy: "pbe0_formation_energy",
        Props.charge: "charge",
        Props.multiplicity: "multiplicity",
        Props.atomic_numbers: "atomic_numbers",
        Props.forces: "pbe0_forces",
        Props.positions: "positions",
        Props.dipole: "pbe0_dipole",
    }
)


def get_qm7x_pbe0_dataset(
    rank,
    data_dir,
    dataset_name,
    splits=None,
    dataset_version="1.0.0",
    copy_to_temp=False,
) -> DatasetSplits:
    if splits is None:
        splits = {"train": "train", "val": "val", "test": "test"}
    data_path = Path(data_dir) / dataset_name / dataset_version
    splits = {Split[k]: v for k, v in splits.items()}

    if copy_to_temp:
        data_path_temp = Path("/temp_data") / dataset_name / dataset_version

        if rank == 0:
            import shutil

            if not data_path_temp.exists():
                logger.info(f"Copying data to {data_path_temp} for faster I/O")
                shutil.copytree(data_path, data_path_temp)
            else:
                logger.info(f"Data already exists at {data_path_temp}")
        data_path = data_path_temp

    if dist.is_initialized():
        dist.barrier()  # Ensure data is copied before proceeding

    decoders = {
        "smiles": tfds.decode.SkipDecoding(),
        "smiles_hash": tfds.decode.SkipDecoding(),
    }

    builder = tfds.builder_from_directory(data_path)

    datasets = {k: builder.as_data_source(split=v, decoders=decoders) for k, v in splits.items()}

    return DatasetSplits(
        splits=datasets,
        dataset_props=qm7x_pbe0_props,
    )
