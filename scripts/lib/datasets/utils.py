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

def convert_force(force_value, from_unit, to_unit="Hartree/Bohr"):
    """
    Convert a force value from a given 'from_unit' to the specified 'to_unit'.
    Valid to_unit options here are 'Hartree/Bohr' or 'Hartree/Å'.
    
    Parameters
    ----------
    force_value : float
        Numerical value of the force to be converted.
    from_unit : str
        Unit of the input force. Examples:
          - 'eV/Å'
          - 'kcal/(mol·Å)'
          - 'kJ/(mol·Å)'
          - 'Hartree/Bohr'
          - 'Hartree/Å'
    to_unit : str
        Desired output force unit. 
        Options: 'Hartree/Bohr' or 'Hartree/Å'
    
    Returns
    -------
    float
        Force in the target to_unit.
    """
    
    # First convert from the input unit to "Hartree/Bohr" as an internal standard.
    conversion_to_hartree_bohr = {
        "eV/Å":           0.019447,      
        "kcal/(mol·Å)":   0.000844,
        "kJ/(mol·Å)":     0.000202,
        "Hartree/Bohr":   1.0,
        "Hartree/Å":      1.0 / 0.52917721067,  # 1 / (Bohr in Å)
    }
    
    if from_unit not in conversion_to_hartree_bohr:
        raise ValueError(f"Input unit '{from_unit}' not recognized.")
    
    # Convert input to "Hartree/Bohr"
    force_in_ha_bohr = force_value * conversion_to_hartree_bohr[from_unit]
    
    # Then if the desired output is "Hartree/Bohr," we are done.
    if to_unit == "Hartree/Bohr":
        return force_in_ha_bohr
    
    # If the desired output is "Hartree/Å," we need to multiply by Bohr->Å
    elif to_unit == "Hartree/Å":
        # 1 Bohr = 0.52917721067 Å
        bohr_in_angstrom = 0.52917721067
        return force_in_ha_bohr / bohr_in_angstrom
    
    else:
        raise ValueError(f"Output unit '{to_unit}' not supported.")

