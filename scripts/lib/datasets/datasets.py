from pathlib import Path

import numpy as np
from lib.types import Property as Props
from torch.utils.data import Dataset


class NPZDataset(Dataset):
    def __init__(self, file: Path, props: dict[Props, str]) -> None:
        self.file = file
        self.data = {}  # Stores data loaded from file
        self.len = 0
        self.props = props

        with np.load(file, allow_pickle=True) as npz_file:
            len_data = npz_file[props[Props.forces]].shape[0]
            self.len = len_data
            self.file_data.append({v: npz_file[v] for v in props.values()})

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> dict:
        sample = {
            self.props[Props.energy]: self.data[Props.energy][idx],
            self.props[Props.forces]: self.data[Props.forces][idx],
            self.props[Props.positions]: self.data[Props.positions][idx],
            self.props[Props.atomic_numbers]: self.data[Props.atomic_numbers][idx],
        }
        return sample
