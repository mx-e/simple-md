from pathlib import Path

import numpy as np
from frozendict import frozendict
from lib.datasets.utils import convert_coordinates, non_overlapping_train_test_val_split_hash_based, convert_force
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from load_atoms import AtomsDataset, load_dataset
from sklearn.preprocessing import OrdinalEncoder
from torch import distributed as dist
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


class LoadAtomsDataset(Dataset):
    def __init__(self, atoms_db: AtomsDataset, data_props: Props, group: np.array, force_unit: str) -> None:
        self.atoms_db = atoms_db
        self.data_props = data_props  # Stores data loaded from file
        self.group = group
        self.force_unit = force_unit

    def __len__(self) -> int:
        return len(self.atoms_db)

    def __getitem__(self, idx) -> dict:
        structure = self.atoms_db[idx]
        sample = {}
        for k, v in self.data_props.items():
            if k == Props.forces:
                sample[v] = convert_force(structure.arrays["forces"], from_unit=self.force_unit, to_unit="Hartree/Bohr")
            elif k == Props.atomic_numbers:
                sample[v] = structure.arrays["numbers"]
            elif k == Props.positions:
                sample[v] = convert_coordinates(structure.arrays["positions"], from_unit="Å", to_unit="Bohr")
            elif k == Props.energy:
                sample[v] = structure.info["energy"]
            elif k == Props.dipole:
                sample[v] = structure.info["dipole"]

        return sample


def get_anix_dataset(
    rank: int,
    data_dir: Path,
    molecule_name: str,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> DatasetSplits:
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}
    data_path = data_dir / "anix"

    anix_props = frozendict(
        {
            Props.energy: Props.energy,
            Props.atomic_numbers: Props.atomic_numbers,
            Props.forces: Props.forces,
            Props.positions: Props.positions,
            Props.dipole: Props.dipole,
        }
    )

    if rank == 0:
        load_dataset("ANI-1x", root=data_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    dataset = load_dataset("ANI-1x", root=data_path)

    names = [structure.get_chemical_formula() for structure in dataset]
    # encode string names to unique integers
    molecule_names = np.array(names)
    molecule_ids = OrdinalEncoder().fit_transform(np.array(names).reshape(-1, 1)).reshape(-1)
    dataset = LoadAtomsDataset(dataset, anix_props, molecule_ids, force_unit="eV/Å")

    train_idx, test_idx, val_idx = non_overlapping_train_test_val_split_hash_based(splits, molecule_names, seed=seed)

    datasets = {
        Split.train: Subset(dataset, train_idx),
        Split.val: Subset(dataset, val_idx),
        Split.test: Subset(dataset, test_idx),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=anix_props,
    )


def get_rMD17_dataset(
    rank: int,
    data_dir: Path,
    molecule_name: str,
    splits: dict[str, float] | None = None,
    seed: int = 40,
) -> DatasetSplits:
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}
    data_path = data_dir / "rMD17"

    molecule_name_formula = frozendict(
    {
        "aspirin": "C9H8O4",
        "azobenzene": "C12H10N2",
        "benzene": "C6H6",
        "ethanol": "C2H6O",
        "malonaldehyde": "C3H2O4",
        "naphthalene": "C10H8",
        "paracetamol": "C8H9NO2",
        "salicylic_acid": "C7H6O3",
        "toluene": "C7H8",
        "uracil": "C4H4N2O2",
    }
)

    rmd17_props = frozendict(
        {
            Props.energy: Props.energy,
            Props.atomic_numbers: Props.atomic_numbers,
            Props.forces: Props.forces,
            Props.positions: Props.positions,
        }
    )

    if rank == 0:
        load_dataset("rMD17", root=data_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    dataset = load_dataset("rMD17", root=data_path)
    dataset = dataset.filter_by(
        lambda structure: structure.get_chemical_formula() == molecule_name_formula[molecule_name]
    )

    # encode string names to unique integers
    dataset = LoadAtomsDataset(dataset, rmd17_props, [], force_unit="eV/Å")

    index_array = np.arange(len(dataset))
    train_val, test_idx = train_test_split(index_array, test_size=splits["train"] + splits["val"], random_state=seed)
    train_idx, val_idx = train_test_split(train_val, test_size=splits["val"], random_state=seed)

    datasets = {
        Split.train: Subset(dataset, train_idx),
        Split.val: Subset(dataset, val_idx),
        Split.test: Subset(dataset, test_idx),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=rmd17_props,
    )
