from pathlib import Path

import h5py
import numpy as np
import pytest
from lib.datasets.qm7x import (
    ASEAtomsDBDataset,
    _download_data,
    _download_duplicates_ids,
    calculate_md5,
    create_ase_db_from_qm7x,
    download_and_check,
    extract_xz,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create and return a temporary directory for test data"""
    return tmp_path


def test_qm7x_download(data_dir: Path) -> None:
    """Test downloading QM7X data files"""
    file_id = "1000"
    checksum = "b50c6a5d0a4493c274368cf22285503e"

    # Download file
    tar_path = data_dir / f"{file_id}.xz"
    url = f"https://zenodo.org/record/4288677/files/{file_id}.xz"
    download_and_check(url, tar_path, checksum)

    # Verify file exists and has correct checksum
    assert tar_path.exists()
    assert calculate_md5(tar_path) == checksum

    # Test extraction
    hdf5_path = data_dir / f"{file_id}.hdf5"
    extract_xz(tar_path, hdf5_path)
    assert hdf5_path.exists()


def test_duplicates_download(data_dir: Path) -> None:
    """Test downloading and parsing duplicates file"""
    duplicates_ids = _download_duplicates_ids(data_dir)

    # Verify we got some duplicates
    assert len(duplicates_ids) > 0
    # Check format of IDs (they should be strings without .xyz extension)
    assert all(isinstance(id_, str) for id_ in duplicates_ids)
    assert not any(id_.endswith(".xyz") for id_ in duplicates_ids)


def test_database_creation_and_loading(data_dir: Path) -> None:
    """Test creating and loading ASE database from QM7X data"""
    # Download a single HDF5 file for testing
    downloaded_files = _download_data(data_dir)
    assert len(downloaded_files) > 0

    # Get duplicates
    duplicates_ids = _download_duplicates_ids(data_dir)

    # Create database
    db_path = data_dir / "test.db"
    create_ase_db_from_qm7x(
        hdf5_files=[Path(f) for f in downloaded_files], db_path=db_path, duplicates_ids=duplicates_ids
    )

    # Verify database exists
    assert db_path.exists()

    # Load dataset
    dataset = ASEAtomsDBDataset(db_path)

    try:
        # Basic dataset checks
        assert len(dataset) > 0

        # Check first item
        first_item = dataset[0]
        assert "Z" in first_item
        assert "positions" in first_item
        assert "n_atoms" in first_item

        # Check types and shapes
        assert isinstance(first_item["Z"], np.ndarray)
        assert isinstance(first_item["positions"], np.ndarray)
        assert isinstance(first_item["n_atoms"], np.ndarray)

        assert first_item["positions"].shape[1] == 3  # 3D coordinates
        assert len(first_item["Z"]) == first_item["n_atoms"][0]  # number of atoms matches
        assert first_item["positions"].shape[0] == first_item["n_atoms"][0]  # positions match number of atoms

        # Test that duplicate structures were properly excluded
        def get_num_structures(hdf5_file) -> int:
            count = 0
            with h5py.File(hdf5_file, "r") as f:
                for mol in f.values():
                    count += len(mol)
            return count

        total_structures = sum(get_num_structures(f) for f in downloaded_files)
        assert len(dataset) < total_structures  # Should have fewer structures after duplicate removal

    finally:
        # Clean up dataset connection
        del dataset


def test_error_handling(data_dir: Path) -> None:
    """Test error handling in the data processing pipeline"""
    # Test invalid checksum
    with pytest.raises(RuntimeError, match="Checksum.*does not match"):
        download_and_check("https://zenodo.org/record/4288677/files/1000.xz", data_dir / "1000.xz", "invalid_checksum")

    # Test invalid database path
    with pytest.raises(AssertionError, match="Database path must have .db extension"):
        create_ase_db_from_qm7x(hdf5_files=[Path("dummy.hdf5")], db_path=Path("invalid_path.txt"), duplicates_ids=[])
