import tempfile
from pathlib import Path
import numpy as np
from lib.datasets.ko2020 import dataset_filenames, get_ko2020_dataset, ko2020_props
from loguru import logger


def test_multiple_ko2020_mols(tmp_path: Path) -> None:
    # Test molecules with their expected number of atoms
    test_molecules = {
        "Ag_cluster": 3,  # 3 silver atoms
        "AuMgO": 110,  # Au, Mg, O atoms
        "Carbon_chain": [12, 13],  # variable number of atoms
        "NaCl": [16, 17],  # variable number of atoms
    }

    # Dictionary to store results
    results = {}
    for molecule, expected_atoms in test_molecules.items():
        logger.info(f"\nTesting {molecule} (expected atoms: {expected_atoms})")

        # Create NPZ directory
        npz_dir = tmp_path / "ko2020"
        npz_dir.mkdir(parents=True, exist_ok=True)

        # Download and convert dataset
        get_ko2020_dataset(0, tmp_path, molecule)

        # Verify file exists and load
        mol_path = npz_dir / dataset_filenames[molecule]
        assert mol_path.exists(), f"NPZ file not created for {molecule}"

        # Load and verify contents
        dataset = np.load(mol_path, allow_pickle=True)

        # Check keys
        assert all(key in dataset for key in ko2020_props.values()), f"Missing required keys for {molecule}"

        # Get actual number of atoms for this configuration
        positions = dataset["positions"]
        if positions.dtype == np.dtype("O"):
            atoms_per_config = [pos.shape[0] for pos in positions]
            if len(set(atoms_per_config)) == 1:
                # All configurations have same number of atoms
                n_atoms_actual = atoms_per_config[0]
            else:
                # Truly ragged data
                n_atoms_actual = (min(atoms_per_config), max(atoms_per_config))
        else:  # regular array
            n_atoms_actual = positions.shape[1]

        # Verify number of atoms matches expected
        if isinstance(expected_atoms, list):
            # Convert everything to min/max comparison
            actual_min = n_atoms_actual[0] if isinstance(n_atoms_actual, tuple) else n_atoms_actual
            actual_max = n_atoms_actual[1] if isinstance(n_atoms_actual, tuple) else n_atoms_actual
            assert min(expected_atoms) <= actual_min and actual_max <= max(expected_atoms), (  # noqa: PT018
                f"Atom count {n_atoms_actual} outside expected range {expected_atoms} for {molecule}"
            )
        else:
            assert n_atoms_actual == expected_atoms, (
                f"Wrong number of atoms for {molecule}: expected {expected_atoms}, got {n_atoms_actual}"
            )

        # Store and log results
        num_configs = len(dataset["energy"])
        energy_range = (float(dataset["energy"].min()), float(dataset["energy"].max()))
        atomic_numbers = dataset["atomic_numbers"].tolist()
        charges = dataset["charges"]
        charge_range = (float(charges.min()), float(charges.max()))

        results[molecule] = {
            "num_configs": num_configs,
            "energy_range": energy_range,
            "atomic_numbers": atomic_numbers,
            "num_atoms": n_atoms_actual,  # This now might be a tuple for ragged arrays
            "charge_range": charge_range,
        }

        logger.info(f"Configurations: {num_configs}")
        logger.info(f"Energy range: {energy_range}")
        logger.info(f"Atomic numbers: {atomic_numbers}")
        logger.info(f"Number of atoms: {n_atoms_actual}")
        logger.info(f"Charge range: {charge_range}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = test_multiple_ko2020_mols(Path(tmp_dir))
