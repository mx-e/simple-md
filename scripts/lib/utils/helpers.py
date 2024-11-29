import os
import random
from pathlib import Path

import numpy as np
import torch

from hydra.core.hydra_config import HydraConfig
from lib.types import Property as Props
from ase import Atoms
from ase.io import write


def get_hydra_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def export_xyz(data, dir, filename):
    n_atoms = data[Props.mask].sum().long().cpu().item()
    positions = data[Props.positions][:n_atoms].cpu().numpy()
    ## convert to angstrom
    positions = positions * 0.529177249
    atomic_numbers = data[Props.atomic_numbers][:n_atoms].cpu().numpy()

    atoms = Atoms(numbers=atomic_numbers, positions=positions)

    path = Path(dir) / (filename + ".xyz")
    write(path, atoms, format="xyz")
    return path
