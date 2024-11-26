import torch as th
from torch.utils.data import Dataset

from lib.types import Property as Props, property_dtype


class TorchifyDataset(Dataset):
    def __init__(self, base_dataset, property_map: dict[str, Props]):
        self.dataset = base_dataset
        self.property_map = property_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        torch_sample = {}

        for key, prop in self.property_map.items():
            if key in sample:
                val = th.as_tensor(
                    sample[key], dtype=property_dtype[prop], device=self.device
                )
                if val.ndim == 0:  # Handle scalars
                    val = val.unsqueeze(0)
                torch_sample[prop] = val

        return torch_sample
