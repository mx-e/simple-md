import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from tools.preprocess_fixed_splits_qcml import get_split_hash
from loguru import logger


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

    # check if the split sizes are close to the desired values and warn if not
    train_size = len(train_idxs) / len(groups)
    test_size = len(test_idxs) / len(groups)
    val_size = len(val_idxs) / len(groups)
    if abs(train_size - splits["train"]) > 0.01:
        logger.warning(f"Warning: train split size is {train_size:.2f} instead of {splits['train']:.2f}")
    if abs(test_size - splits["test"]) > 0.01:
        logger.warning(f"Warning: test split size is {test_size:.2f} instead of {splits['test']:.2f}")
    if abs(val_size - splits["val"]) > 0.01:
        logger.warning(f"Warning: val split size is {val_size:.2f} instead of {splits['val']:.2f}")

    return train_idxs, test_idxs, val_idxs
