import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def non_overlapping_train_test_val_split(
        splits: dict[str, float],
        index_array: np.array,
        groups: np.array, seed: int=42
    ) -> tuple[np.array, np.array, np.array]:
    gss = GroupShuffleSplit(n_splits=1, test_size=splits["train"] + splits["val"], random_state=0).split(index_array, index_array, groups=groups)
    train_val, test = next(gss)

    gss = GroupShuffleSplit(n_splits=1, test_size=splits["val"], random_state=0).split(train_val, train_val, groups=groups[train_val])
    train_idx, val_idx = next(gss)
    train = train_val[train_idx]
    val = train_val[val_idx]
    return test,train,val