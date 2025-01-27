import gzip
import json
import numpy as np
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from tools.preprocess_fixed_splits_qcml import get_split_hash
from frozendict import frozendict

molecule_name_formula = frozendict(
    {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "azobenzene": "C1=CC=C(C=C1)N=NC2=CC=CC=C2 ",
        "benzene": "C1=CC=CC=C1 ",
        "ethanol": "CCO",
        "malonaldehyde": "C(C=O)C=O ",
        "naphthalene": "C1=CC=C2C=CC=CC2=C1 ",
        "paracetamol": "CC(=O)NC1=CC=C(C=C1)O ",
        "salicylic_acid": "C1=CC=C(C(=C1)C(=O)O)O ",
        "toluene": "CC1=CC=CC=C1 ",
        "uracil": "C1=CNC(=O)NC1=O ",
        "NaCl": "[Na+].[Cl-]",
        "Carbon_chain": "none",
        "AuMgO": "none",
        "ag_cluster": "none",
        "Ac-Ala3-NHMe": "none",
        "DHA": "C1[C@H]([C@H]([C@H](OC1(C(=O)O)O)C(=O)O)O)O",
        "stachyose": "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)OC[C@@H]2[C@@H]([C@@H]([C@H]([C@H](O2)OC[C@@H]3[C@H]([C@@H]([C@H]([C@H](O3)O[C@]4([C@H]([C@@H]([C@H](O4)CO)O)O)CO)O)O)O)O)O)O)O)O)O)O",
        "AT-AT": "none",
        "AT-AT-CG-CG": "none",
        "buckyball-catcher": "none",
        "double-walled_nanotube": "none",
    }
)

def get_split_by_molecule_name(molecule_name: str) -> str:
    if molecule_name not in molecule_name_formula:
        logger.warning(f"Unknown chemical formula for '{molecule_name}'.")
        return "unknown"
    
    # load split from gzipped file
    with gzip.open('data_smiles/smiles.json.gz', 'rb') as f:
        splits = json.load(f)

    #remove all slashes from the molecule names in the splits
    for key in splits.keys():
        splits[key] = [x.replace('/','').replace('\\','') for x in splits[key]]
    
    for key in splits.keys():
        if molecule_name_formula[molecule_name] in splits[key]:
            return key
    return "unknown"

def non_overlapping_train_test_val_split(
    splits: dict[str, float], groups: np.array, seed: int = 42
) -> tuple[np.array, np.array, np.array]:
    index_array = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=splits["train"] + splits["val"], random_state=seed).split(
        index_array, index_array, groups=groups
    )
    train_val, test = next(gss)

    gss = GroupShuffleSplit(n_splits=1, test_size=splits["val"], random_state=seed).split(
        train_val, train_val, groups=groups[train_val]
    )
    train_idx, val_idx = next(gss)
    train = train_val[train_idx]
    val = train_val[val_idx]
    return test, train, val


def non_overlapping_train_test_val_split_hash_based(
    splits: dict[str, float], groups: np.array, seed: int = 42, prevent_warnings: bool = False
) -> tuple[np.array, np.array, np.array]:
    hash_val = get_split_hash(groups, seed).numpy()

    train_idxs = np.where(hash_val < splits["train"])[0]
    test_idxs = np.where((hash_val >= splits["train"]) & (hash_val < splits["train"] + splits["test"]))[0]
    val_idxs = np.where(hash_val >= splits["train"] + splits["test"])[0]

    if not prevent_warnings:
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


def convert_coordinates(positions: np.ndarray, from_unit: str, to_unit: str = "Bohr") -> np.ndarray:
    conversion_to_bohr = {
        "Bohr": 1.0,
        "Å": 1.8897259,
    }

    if from_unit not in conversion_to_bohr:
        raise ValueError(f"Input unit '{from_unit}' not recognized.")

    pos_bohr = positions * conversion_to_bohr[from_unit]

    if to_unit == "Bohr":
        return pos_bohr

    else:
        raise ValueError(f"Output unit '{to_unit}' not supported.")


def convert_force(force_value: float, from_unit: str, to_unit: str = "Hartree/Bohr") -> float:
    """Convert a force value from a given 'from_unit' to the specified 'to_unit'.
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
        "eV/Å": 0.019447,
        "kcal/(mol·Å)": 0.0008432982619,
        "kJ/(mol·Å)": 0.000202,
        "Hartree/Bohr": 1.0,
        "Hartree/Å": 1.0 / 0.52917721067,  # 1 / (Bohr in Å)
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
