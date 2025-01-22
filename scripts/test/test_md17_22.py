import tempfile
from pathlib import Path

import numpy as np
from lib.datasets.md17_22 import _filenames, download_md17_22_dataset, md17_props
from loguru import logger


def test_multiple_md17_mols(tmp_path: Path) -> None:
    # Let's test a subset of molecules with different sizes
    test_molecules = {
        "aspirin": 21,  # 21 atoms
        "benzene": 12,  # 12 atoms
        "ethanol": 9,  # 9 atoms
        "malonaldehyde": 9,  # 9 atoms
        "toluene": 15,  # 15 atoms
    }

    for molecule, n_atoms in test_molecules.items():
        logger.info(f"\nTesting {molecule} (expected atoms: {n_atoms})")

        # Download dataset
        download_md17_22_dataset(tmp_path, molecule)

        # Verify file exists and load
        mol_path = tmp_path / _filenames[molecule]
        assert mol_path.exists(), f"Download failed for {molecule} - file not created"

        # Load and verify contents
        dataset = np.load(mol_path, allow_pickle=True)

        # Check keys
        assert all(key in dataset for key in md17_props.values()), f"Missing required keys for {molecule}"

        # Verify number of atoms
        assert dataset["z"].shape == (n_atoms,), f"Wrong number of atoms for {molecule}"
        assert dataset["F"].shape[1:] == (n_atoms, 3), f"Wrong force shape for {molecule}"
        assert dataset["R"].shape[1:] == (n_atoms, 3), f"Wrong position shape for {molecule}"

        num_configs = len(dataset["E"])
        energy_range = (float(dataset["E"].min()), float(dataset["E"].max()))
        atomic_numbers = np.unique(dataset["z"]).tolist()
        num_atoms = n_atoms

        logger.info(f"Configurations: {num_configs}")
        logger.info(f"Energy range: {energy_range}")
        logger.info(f"Atomic numbers: {atomic_numbers}")
        logger.info(f"Number of atoms: {num_atoms}")


if __name__ == "__main__":
    results = test_multiple_md17_mols()
