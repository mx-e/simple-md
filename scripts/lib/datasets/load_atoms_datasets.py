from pathlib import Path
from load_atoms import load_dataset, AtomsDataset

import numpy as np
from frozendict import frozendict
from lib.types import DatasetSplits, Split
from lib.types import Property as Props
from lib.datasets.utils import non_overlapping_train_test_val_split
from loguru import logger
from torch import distributed as dist
from torch.utils.data import Subset, Dataset
from sklearn.preprocessing import OrdinalEncoder

class LoadAtomsDataset(Dataset):
    def __init__(self, atoms_db: AtomsDataset, data_props: Props, group: np.array) -> None:
        self.atoms_db = atoms_db
        self.data_props = data_props  # Stores data loaded from file
        self.group = group

    def __len__(self) -> int:
        return len(self.atoms_db)

    def __getitem__(self, idx) -> dict:
        structure = self.atoms_db[idx]
        sample = {}
        for k,v in self.data_props.items():
            if k == Props.forces:
                sample[v] = structure.arrays["forces"]
            elif k == Props.atomic_numbers:
                sample[v] = structure.arrays["numbers"]
            elif k == Props.positions:
                sample[v] = structure.arrays["positions"]
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
):
    if splits is None:
        splits = {"train": 0.5, "val": 0.3, "test": 0.2}
    data_path = data_dir / "anix"

    anix_props = frozendict(
        {
            Props.energy: Props.energy,
            Props.atomic_numbers: Props.atomic_numbers,
            Props.forces: Props.forces,
            Props.positions: Props.positions,
            Props.dipole: Props.dipole
        }
    )
    
    if rank == 0:
        load_dataset("ANI-1x", root=data_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    dataset = load_dataset("ANI-1x", root=data_path)

    names = []
    for idx, structure in enumerate(dataset):
        names.append(structure.get_chemical_formula())
    # encode string names to unique integers
    molecule_ids = OrdinalEncoder().fit_transform(np.array(names).reshape(-1, 1)).reshape(-1)
    dataset = LoadAtomsDataset(dataset, anix_props, molecule_ids)

    index_array = np.arange(len(dataset))
    test, train, val = non_overlapping_train_test_val_split(splits, index_array, molecule_ids, seed=seed)


    datasets = {
        Split.train: Subset(dataset, train),
        Split.val: Subset(dataset, val),
        Split.test: Subset(dataset, test),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=anix_props,
    )