from pathlib import Path

import numpy as np
from lib.types import Property as Props
from torch.utils.data import Dataset
from lib.datasets.utils import convert_force


class NPZDataset(Dataset):
    def __init__(self, file_path: Path, props: dict[Props, str], force_unit: str) -> None:
        self.file_path = file_path
        self.data = {}  # Stores data loaded from file
        self.len = 0
        self.props = props
        self.force_unit = force_unit

        with np.load(file_path, allow_pickle=True) as npz_file:
            len_data = npz_file[props[Props.forces]].shape[0]
            self.len = len_data
            self.data = {v: npz_file[v] for v in props.values()}

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> dict:
        sample = {
            self.props[Props.energy]: self.data[self.props[Props.energy]][idx],
            self.props[Props.forces]: convert_force(self.data[self.props[Props.forces]][idx], from_unit=self.force_unit, to_unit="Hartree/Bohr"),
            self.props[Props.positions]: self.data[self.props[Props.positions]][idx],
            self.props[Props.atomic_numbers]: self.data[self.props[Props.atomic_numbers]][idx],
        }
        return sample
