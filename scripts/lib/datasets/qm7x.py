import hashlib
import lzma
import shutil
from pathlib import Path
from urllib import request

import h5py
from ase import Atoms
from ase.db import connect
from frozendict import frozendict
from lib.datasets.utils import non_overlapping_train_test_val_split_hash_based, convert_force
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from loguru import logger
from torch import distributed as dist
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

qm7x_props = frozendict(
    {
        Props.energy: "energy",
        Props.atomic_numbers: "Z",
        Props.forces: "forces",
        Props.positions: "positions",
    }
)

__pbar = None


class ASEAtomsDBDataset(Dataset):
    def __init__(self, db_path: Path, force_unit: str) -> None:
        self.db_path = db_path
        self.conn = connect(self.db_path, use_lock_file=False)

    def __len__(self) -> int:
        return self.conn.count()

    def __getitem__(self, idx) -> dict:
        # ASE DB uses 1-based indexing
        row = self.conn.get(idx + 1)
        Z = row.numbers
        positions = row.positions
        # properties
        properties = row.data

        return {
            "Z": Z,
            "positions": positions,
            "energy": properties["energy"],
            "forces": convert_force(properties["forces"], from_unit="eV/Å", to_unit="Hartree/Bohr"),
        }

    def get_chemical_formula(self, idx) -> str:
        row = self.conn.get(idx + 1)
        return row.data["chemical_formula"]


def get_qm7x_dataset(
    rank: int,
    data_dir: Path,
    workdir: Path | None = None,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> DatasetSplits:
    """Get QM7-X dataset splits.

    Args:
        rank: Process rank for distributed training
        data_dir: Directory where the final dataset should be stored
        workdir: Working directory for downloads and processing. If None, uses data_dir
        splits: Dictionary with train/val/test split ratios. Defaults to 50/30/20
        seed: Random seed for reproducibility

    Returns:
        DatasetSplits object containing the dataset splits and properties
    """
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}

    # Setup directories
    working_path = workdir if workdir is not None else data_dir

    # Create directory structure
    raw_dir = working_path / "qm7x" / "raw"
    db_dir = data_dir / "qm7x"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    permanent_db_path = db_dir / "qm7x.db"
    db_path = (working_path / "qm7x" / "qm7x.db") if workdir else permanent_db_path

    if not permanent_db_path.exists() and rank == 0:
        logger.info(f"QM7-X dataset not found, downloading to {raw_dir}")
        hdf5_files = _download_data(raw_dir)
        duplicates_ids = _download_duplicates_ids(raw_dir)

        create_ase_db_from_qm7x(hdf5_files, db_path, duplicates_ids)

        logger.info("Cleaning up intermediate files...")
        shutil.rmtree(raw_dir)

        if workdir is not None:
            logger.info(f"Copying database to permanent storage: {permanent_db_path}")
            shutil.copy2(db_path, permanent_db_path)
    elif workdir is not None and rank == 0:
        logger.info(f"Copying database to fast storage: {permanent_db_path}")
        shutil.copy2(permanent_db_path, db_path)

    # Wait for rank 0 to finish database operations
    dist.barrier()

    # Create dataset from the database
    dataset = ASEAtomsDBDataset(db_path, force_unit="eV/Å")

    # Split dataset
    molecule_names = [dataset.get_chemical_formula(i) for i in range(len(dataset))]

    train_idx, test_idx, val_idx = non_overlapping_train_test_val_split_hash_based(
        splits, molecule_names, seed
    )

    datasets = {
        Split.train: Subset(dataset, train_idx),
        Split.val: Subset(dataset, val_idx),
        Split.test: Subset(dataset, test_idx),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=qm7x_props,
    )


def create_ase_db_from_qm7x(hdf5_files: list[Path], db_path: Path, duplicates_ids: list[str]) -> None:
    property_keys = {
        "energy": "ePBE0+MBD",  # total energy
        "forces": "totFOR",  # total forces
    }

    assert db_path.suffix == ".db", "Database path must have .db extension"

    logger.info("Creating ASE database from QM7-X data files...")

    # Create database connection
    with connect(db_path) as db:
        # Process each HDF5 file
        for file_path in hdf5_files:
            logger.info(f"Processing {file_path.stem}...")

            with h5py.File(file_path, "r") as f:
                # Iterate over molecules in the file
                for mol in tqdm(f.values()):
                    # Iterate over conformers of each molecule
                    for conf_id, conf in mol.items():
                        # Check for duplicates
                        # Extract the ID without the conformer/step suffix
                        trunc_id = conf_id[::-1].split("-", 1)[-1][::-1]
                        if trunc_id in duplicates_ids:
                            continue

                        # Create ASE Atoms object
                        atoms = Atoms(positions=conf["atXYZ"][:], numbers=conf["atNUM"][:])

                        # Extract requested properties
                        properties = {}
                        for prop_name, qm7x_key in property_keys.items():
                            if qm7x_key in conf:
                                properties[prop_name] = conf[qm7x_key][:]

                        # add sum formula to properties
                        properties["chemical_formula"] = atoms.get_chemical_formula()

                        # Write to database
                        db.write(atoms, data=properties)

    logger.info(f"Database created at {db_path}")
    logger.info(f"Total entries: {db.count()}")


def _download_duplicates_ids(tar_dir: Path) -> None:
    url = "https://zenodo.org/record/4288677/files/DupMols.dat"
    tar_path = tar_dir / "DupMols.dat"
    checksum = "5d886ccac38877c8cb26c07704dd1034"

    download_and_check(url, tar_path, checksum)
    duplicates_ids = [line.rstrip("\n")[:-4] for line in tar_path.open("r")]
    return duplicates_ids


def _download_data(tar_dir: Path) -> list[str]:
    file_ids = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000"]

    # file fingerprints to check integrity
    checksums = [
        "b50c6a5d0a4493c274368cf22285503e",
        "4418a813daf5e0d44aa5a26544249ee6",
        "f7b5aac39a745f11436047c12d1eb24e",
        "26819601705ef8c14080fa7fc69decd4",
        "85ac444596b87812aaa9e48d203d0b70",
        "787fc4a9036af0e67c034a30ad854c07",
        "5ecce00a188410d06b747cb683d8d347",
        "c893ae88b8f5c32541c3f024fc1daa45",
    ]

    logger.info("Downloading QM7-X data files ...")

    # download and extract files
    for i, file_id in enumerate(file_ids):
        url = f"https://zenodo.org/record/4288677/files/{file_id}.xz"
        tar_path = tar_dir / f"{file_id}.xz"
        download_and_check(url, tar_path, checksums[i])
    extracted = []
    for file_id in file_ids:
        xz_path = tar_dir / f"{file_id}.xz"
        hd_path = tar_dir / f"{file_id}.hdf5"
        extract_xz(xz_path, hd_path)
        extracted.append(hd_path)
    return extracted


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    global __pbar  # noqa: PLW0603
    if __pbar is None:
        __pbar = tqdm(total=total_size, unit="B", unit_scale=True)

    downloaded = block_num * block_size
    if downloaded < total_size:
        # Update with the increment since last iteration
        __pbar.update(block_size)
    else:
        __pbar.close()
        __pbar = None


def calculate_md5(filepath: Path) -> str:
    return hashlib.md5(filepath.read_bytes()).hexdigest()  # noqa: S324


def download_and_check(url: str, tar_path: Path, checksum: str) -> None:
    file = Path(url).name
    if tar_path.exists():
        md5_sum = calculate_md5(tar_path)
        if md5_sum == checksum:
            logger.info(f"File {file} already exists and has correct checksum. Skipping download.")
            return
        else:
            logger.info(f"File {file} already exists but has wrong checksum. Redownloading.")
            tar_path.unlink()

    logger.info(f"Downloading {url} ...")
    request.urlretrieve(url, tar_path, show_progress)  # noqa: S310

    # Verify downloaded file
    if calculate_md5(tar_path) != checksum:
        tar_path.unlink()  # Clean up invalid file
        raise RuntimeError(f"Checksum of downloaded file {file} does not match. Please try again.")


def extract_xz(source: Path, target: Path) -> None:
    s_file = source.name
    t_file = target.name

    if target.exists():
        logger.info(f"File {t_file} already exists. Skipping extraction.")
        return

    logger.info(f"Extracting {s_file} ...")
    try:
        with lzma.open(source) as fin, target.open(mode="wb") as fout:
            shutil.copyfileobj(fin, fout)
    except Exception as e:
        if target.exists():
            target.unlink()
        raise RuntimeError(f"Could not extract file {s_file}. Please try again.") from e

    logger.info("Done.")

get_qm7x_dataset(0, Path("."), Path("."))