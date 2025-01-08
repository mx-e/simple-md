import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from tools.preprocess_fixed_splits_qcml import get_split_hash


def non_overlapping_train_test_val_split(
        splits: dict[str, float],
        groups: np.array, seed: int=42
    ) -> tuple[np.array, np.array, np.array]:
    index_array = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=splits["train"] + splits["val"], random_state=seed).split(index_array, index_array, groups=groups)
    train_val, test = next(gss)

    gss = GroupShuffleSplit(n_splits=1, test_size=splits["val"], random_state=seed).split(train_val, train_val, groups=groups[train_val])
    train_idx, val_idx = next(gss)
    train = train_val[train_idx]
    val = train_val[val_idx]
    return test,train,val

def non_overlapping_train_test_val_split_hash_based(
        splits: dict[str, float],
        groups: np.array, seed: int=42
    ) -> tuple[np.array, np.array, np.array]:

    hash_val = get_split_hash(groups, seed).numpy()

    train_idxs = np.where(hash_val < splits["train"])[0]
    test_idxs = np.where((hash_val >= splits["train"]) & (hash_val < splits["train"] + splits["test"]))[0]
    val_idxs = np.where(hash_val >= splits["train"] + splits["test"])[0]

    return train_idxs, test_idxs, val_idxs
