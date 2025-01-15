from pathlib import Path
from typing import Dict, List

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.db import connect
from loguru import logger
import requests
import shutil
import tarfile


def read_runner_configurations(file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with file_path.open() as f:
        content = f.read()

    # Split into configurations
    configs = [block.strip() for block in content.split("begin") if block.strip()]
    configs = [block.split("end")[0].strip() for block in configs]

    # Initialize lists to store data
    all_positions = []
    all_forces = []
    all_energies = []
    all_charges = []  # Added for molecular charges
    atomic_nums = None

    for config in configs:
        if not config:
            continue

        positions = []
        forces = []
        energy = None
        charge = None  # For molecular charge
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

        if atomic_nums is None:
            atomic_nums = np.array([atomic_numbers[sym] for sym in symbols])

        all_positions.append(positions)
        all_forces.append(forces)
        all_energies.append(energy)
        all_charges.append(charge)

    # Convert to numpy arrays
    positions_array = np.array(all_positions)  # Shape: (n_configs, n_atoms, 3)
    forces_array = np.array(all_forces)  # Shape: (n_configs, n_atoms, 3)
    energies_array = np.array(all_energies)  # Shape: (n_configs,)
    charges_array = np.array(all_charges)  # Shape: (n_configs,)

    return positions_array, forces_array, energies_array, atomic_nums, charges_array


def convert_runner_to_npz(source_path: Path, target_path: Path) -> None:
    # Read configurations
    positions, forces, energies, atomic_numbers, charges = read_runner_configurations(source_path)

    # Save as NPZ file
    np.savez(
        target_path,
        positions=positions,
        forces=forces,
        energy=energies,
        atomic_numbers=atomic_numbers,
        charges=charges,
    )

    # Print summary
    logger.info(f"Converted {source_path} to {target_path}")
    logger.info(f"Dataset contains {len(energies)} configurations")
    logger.info("Data shapes:")
    logger.info(f"  Positions: {positions.shape}")
    logger.info(f"  Forces: {forces.shape}")
    logger.info(f"  Energies: {energies.shape}")
    logger.info(f"  Atomic numbers: {atomic_numbers.shape}")
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


def get_ko2020_dataset(
    rank: int,
    data_dir: Path,
    workdir: Path | None = None,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> None:
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}

    working_path = workdir if workdir is not None else data_dir

    raw_dir = working_path / "ko2020" / "raw"
    db_dir = data_dir / "ko2020"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    permanent_db_path = db_dir / "ko2020.db"
