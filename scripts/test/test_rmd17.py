import tempfile
from pathlib import Path

import numpy as np
from lib.datasets.rmd17 import _filenames, download_rmd17_dataset, rmd17_props
from lib.types import Property as Props
from loguru import logger


def test_multiple_md17_mols(tmp_path: Path) -> dict:
    # Let's test a subset of molecules with different sizes
    test_molecules = {
        "aspirin": 21,  # 21 atoms
        "benzene": 12,  # 12 atoms
        "ethanol": 9,  # 9 atoms
        "malonaldehyde": 9,  # 9 atoms
        "toluene": 15,  # 15 atoms
    }

    download_rmd17_dataset(tmp_path)
    for molecule, n_atoms in test_molecules.items():
        logger.info(f"\nTesting {molecule} (expected atoms: {n_atoms})")

        # Verify file exists and load
        mol_path = tmp_path / "rmd17" / "npz_data" / _filenames[molecule]
        assert mol_path.exists(), f"Download failed for {molecule} - file not created"

        # Load and verify contents
        dataset = np.load(mol_path, allow_pickle=True)
        logger.info(f"keys: {list(dataset.keys())}")
        # Check keys
        assert all(key in dataset for key in rmd17_props.values()), f"Missing required keys for {molecule}"

        # Verify number of atoms
        assert dataset[rmd17_props[Props.atomic_numbers]].shape == (n_atoms,), f"Wrong number of atoms for {molecule}"
        assert dataset[rmd17_props[Props.forces]].shape[1:] == (n_atoms, 3), f"Wrong force shape for {molecule}"
        assert dataset[rmd17_props[Props.positions]].shape[1:] == (
            n_atoms,
            3,
        ), f"Wrong position shape for {molecule}"

        energies = dataset[rmd17_props[Props.energy]]
        num_configs = len(energies)
        energy_range = (float(energies.min()), float(energies.max()))
        atomic_numbers = np.unique(dataset[rmd17_props[Props.atomic_numbers]]).tolist()
        num_atoms = n_atoms

        logger.info(f"Configurations: {num_configs}")
        logger.info(f"Energy range: {energy_range}")
        logger.info(f"Atomic numbers: {atomic_numbers}")
        logger.info(f"Number of atoms: {num_atoms}")


if __name__ == "__main__":
    results = test_multiple_md17_mols()
