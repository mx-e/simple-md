import os
import random
from pathlib import Path

import numpy as np
import torch

from hydra.core.hydra_config import HydraConfig


def get_hydra_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
