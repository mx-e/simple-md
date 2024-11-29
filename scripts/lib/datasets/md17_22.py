import os
import numpy as np
import tensorflow_datasets as tfds
import torch.distributed as dist
from loguru import logger
from frozendict import frozendict
from pathlib import Path
from scripts.lib.types import Property as Props, DatasetSplits, Split
from copy import deepcopy

from sklearn.model_selection import train_test_split

md17_22_props = frozendict(
    {
        Props.energy: "E",
        Props.atomic_numbers: "z",
        Props.forces: "F",
        Props.positions: "R",
    }
)

import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


class NPZDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        Args:
            file_list (list of str): List of paths to .npz files.
            transform (callable, optional): Optional transform to apply to data.
        """
        self.file_list = file_list
        self.transform = transform
        
        self.file_data = []  # Stores data loaded from each file
        self.file_indices = []    # Maps dataset indices to (file_idx, row_idx)
        self.dataset_idx_mapping = {}  # Maps dataset indices to (file_idx, row_idx)

        file_idx = 0
        dataset_len = 0
        # Load metadata and construct index map
        # keep a dict of dataset index to (file_idx, row_idx) mapping
        for file_path in file_list:
            if file_path.endswith('.npz'):
                with np.load(file_path, allow_pickle=True) as npz_file:
                    len_data = npz_file['F'].shape[0]
                    file_indices = [file_idx]*len_data
                    self.file_indices.extend(file_indices)
                    self.file_data.append({v: npz_file[v] for v in md17_22_props.values()})
                    self.dataset_idx_mapping.update({dataset_len + i: (file_idx, i) for i in range(len_data)})
                    file_idx += 1
                    dataset_len += len_data

        self.file_indices = np.array(self.file_indices, dtype=int)

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx, row_idx = self.dataset_idx_mapping[idx]
        
        sample = {
            "E": self.file_data[file_idx]["E"][row_idx],
            "F": self.file_data[file_idx]["F"][row_idx],
            "R": self.file_data[file_idx]["R"][row_idx],
            "z": self.file_data[file_idx]["z"],
        }
        
        return sample



def get_md17_22_dataset(
    rank,
    data_dir,
    dataset_name,
    splits={"train": 0.5, "val": 0.3, "test": 0.2},
    seed=42,
    **kwargs,
):

    data_path = os.path.join(data_dir, dataset_name)
    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]
    dataset = NPZDataset(file_list)
    print(f"Dataset length: {len(dataset)}")
    index_array = np.arange(len(dataset))
    train_val, test = train_test_split(
        index_array,
        test_size=splits["train"] + splits["val"],
        random_state=seed,
        stratify=dataset.file_indices
    )
    train, val = train_test_split(
        train_val,
        test_size=splits["val"],
        random_state=seed,
        stratify=dataset.file_indices[train_val]
    )

    datasets = {
        Split.train: Subset(dataset, train),
        Split.val: Subset(dataset, val),
        Split.test: Subset(dataset, test),
    }

    return DatasetSplits(
        splits=datasets,
        dataset_props=md17_22_props,
    )
