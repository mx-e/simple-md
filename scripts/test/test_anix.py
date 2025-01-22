import pytest
from pathlib import Path
from lib.datasets.load_atoms_datasets import get_anix_dataset
from lib.types import Split, Property as Props


def test_get_anix_dataset(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    dataset_splits = get_anix_dataset(0, data_dir, None)

    assert len(dataset_splits.splits[Split.train]) > 0
    assert len(dataset_splits.splits[Split.val]) > 0
    assert len(dataset_splits.splits[Split.test]) > 0

    assert dataset_splits.dataset_props[Props.energy] == Props.energy
    assert dataset_splits.dataset_props[Props.atomic_numbers] == Props.atomic_numbers
    assert dataset_splits.dataset_props[Props.forces] == Props.forces
    assert dataset_splits.dataset_props[Props.positions] == Props.positions
    assert dataset_splits.dataset_props[Props.dipole] == Props.dipole

    # Add more assertions as needed


if __name__ == "__main__":
    pytest.main([__file__])
