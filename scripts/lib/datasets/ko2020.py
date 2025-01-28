import shutil
import tarfile
from pathlib import Path

import numpy as np
import requests
from ase.data import atomic_numbers
from frozendict import frozendict
from lib.datasets.datasets import NPZDataset
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from loguru import logger
from sklearn.model_selection import train_test_split
from scripts.lib.datasets.utils import get_split_by_molecule_name
from torch import distributed as dist
from torch.utils.data import Subset

ko2020_props = frozendict(
    {
        Props.energy: "energy",
        Props.forces: "forces",
        Props.positions: "positions",
        Props.atomic_numbers: "atomic_numbers",
        Props.charge: "charges",
    }
)


def read_runner_configurations(file_path: Path) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    with file_path.open() as f:
        content = f.read()

    # Split into configurations
    configs = [block.strip() for block in content.split("begin") if block.strip()]
    configs = [block.split("end")[0].strip() for block in configs]

    # Initialize lists to store data
    all_positions = []
    all_forces = []
    all_energies = []
    all_charges = []
    all_atomic_nums = []

    for config in configs:
        if not config:
            continue

        positions = []
        forces = []
        energy = None
        charge = None
        symbols = []

        for line in config.split("\n"):
            parts = line.split()
            if parts[0] == "atom":
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                symbols.append(parts[4])
                if len(parts) > 8:
                    forces.append([float(parts[7]), float(parts[8]), float(parts[9])])
            elif parts[0] == "energy":
                energy = float(parts[1])
            elif parts[0] == "charge":
                charge = float(parts[1])

        atomic_nums = np.array([atomic_numbers[sym] for sym in symbols])

        all_atomic_nums.append(atomic_nums)
        all_positions.append(np.array(positions))
        all_forces.append(np.array(forces))
        all_energies.append(energy)
        all_charges.append(charge)

    # Convert to arrays where appropriate (not for ragged arrays)
    energies_array = np.array(all_energies)
    charges_array = np.array(all_charges)

    logger.info(
        f"Number of atoms per molecule: min={min(len(pos) for pos in all_positions)}, "
        f"max={max(len(pos) for pos in all_positions)}"
    )

    return all_positions, all_forces, energies_array, all_atomic_nums, charges_array


def convert_runner_to_npz(source_path: Path, target_path: Path) -> None:
    # Read configurations
    positions, forces, energies, atomic_numbers, charges = read_runner_configurations(source_path)

    # Convert to object arrays for ragged data
    positions_array = np.array(positions, dtype=object)
    forces_array = np.array(forces, dtype=object)
    atomic_nums_array = np.array(atomic_numbers, dtype=object)

    # Save as NPZ file
    np.savez(
        target_path,
        positions=positions_array,
        forces=forces_array,
        energy=energies,
        atomic_numbers=atomic_nums_array,
        charges=charges,
    )

    # Print summary
    logger.info(f"Converted {source_path} to {target_path}")
    logger.info(f"Dataset contains {len(energies)} configurations")
    logger.info("Data shapes:")
    logger.info(f"  Positions: array of shape ({len(positions)},) containing arrays of shape (n_atoms, 3)")
    logger.info(f"  Forces: array of shape ({len(forces)},) containing arrays of shape (n_atoms, 3)")
    logger.info(f"  Energies: {energies.shape}")
    logger.info(f"  Atomic numbers: {len(atomic_numbers)} arrays of shape (n_atoms,)")
    logger.info(f"  Molecular charges: {charges.shape}")


def download_and_extract_dataset(raw_folder: Path) -> None:
    raw_folder.mkdir(parents=True, exist_ok=True)
    url = "https://archive.materialscloud.org/record/file?record_id=629&filename=datasets.tar.gz"

    logger.info("Downloading dataset...")

    response = requests.get(url, stream=True)  # noqa: S113
    response.raise_for_status()

    tar_path = raw_folder / "datasets.tar.gz"
    with tar_path.open("wb") as f:
        shutil.copyfileobj(response.raw, f)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=raw_folder)  # noqa: S202

    # Remove the tar file after extraction
    tar_path.unlink()

    logger.info(f"Dataset downloaded and extracted to {raw_folder}")


dataset_rel_paths = frozendict(
    {
        "Ag_cluster": "datasets/Ag_cluster/input.data",
        "AuMgO": "datasets/AuMgO/input.data",
        "Carbon_chain": "datasets/Carbon_chain/input.data",
        "NaCl": "datasets/NaCl/input.data",
    }
)

dataset_filenames = frozendict(
    {
        "Ag_cluster": "Ag_cluster.npz",
        "AuMgO": "AuMgO.npz",
        "Carbon_chain": "Carbon_chain.npz",
        "NaCl": "NaCl.npz",
    }
)


def get_ko2020_dataset(
    rank: int,
    data_dir: Path,
    molecule_name: str,
    work_dir: Path | None = None,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> None:
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}

    assert molecule_name in dataset_filenames, (
        f"Unknown molecule {molecule_name=}, expected one of {dataset_filenames.keys()}"
    )

    working_path = work_dir if work_dir is not None else data_dir
    npz_dir = data_dir / "ko2020"
    npz_file_path = npz_dir / dataset_filenames[molecule_name]
    out_file_paths = {k: npz_dir / v for k, v in dataset_filenames.items()}

    if not npz_file_path.exists() and rank == 0:
        logger.info(f"KO2020-{molecule_name} not found, downloading to {working_path}")
        raw_dir = working_path / "ko2020" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        npz_dir.mkdir(parents=True, exist_ok=True)

        download_and_extract_dataset(raw_dir)
        raw_file_paths = {k: raw_dir / v for k, v in dataset_rel_paths.items()}
        for mol, raw_path in raw_file_paths.items():
            convert_runner_to_npz(raw_path, out_file_paths[mol])

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    dataset = NPZDataset(out_file_paths[molecule_name], ko2020_props, force_unit="kcal/(mol·Å)", coord_unit="Å")

    # get split in which this molecule is probably included during training
    split_name = get_split_by_molecule_name(molecule_name)
    logger.info(f"This molecule was included in {'the '+ split_name if split_name != "unknown" else 'no '} split during training.")

    index_array = np.arange(len(dataset))
    train_idx, test_val_idx = train_test_split(index_array, train_size=splits["train"], random_state=seed)
    test_idx, val_idx = train_test_split(test_val_idx, train_size=splits["test"], random_state=seed)

    datasets = {
        Split.train: Subset(dataset, train_idx),
        Split.val: Subset(dataset, val_idx),
        Split.test: Subset(dataset, test_idx),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=ko2020_props,
    )
