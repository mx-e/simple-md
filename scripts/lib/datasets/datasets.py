from pathlib import Path

import numpy as np
from lib.datasets.utils import convert_force, convert_coordinates
from lib.types import Property as Props
from torch.utils.data import Dataset


from pathlib import Path
import numpy as np
from lib.datasets.utils import convert_force, convert_coordinates
from lib.types import Property as Props
from torch.utils.data import Dataset
from typing import Any


class NPZDataset(Dataset):
    def __init__(self, file_path: Path, props: dict[Props, str], force_unit: str, coord_unit: str) -> None:
        self.file_path = file_path
        self.data: dict[str, Any] = {}  # Stores data loaded from file
        self.len = 0
        self.props = props
        self.force_unit = force_unit
        self.coord_unit = coord_unit
        self.is_ragged = False

        # Load data with allow_pickle=True to support both regular and object arrays
        with np.load(file_path, allow_pickle=True) as npz_file:
            # Load the forces first to determine if we have ragged arrays
            forces = npz_file[props[Props.forces]]
            self.is_ragged = forces.dtype == np.dtype("O")

            # Determine length based on array type
            if self.is_ragged:
                self.len = len(forces)
            else:
                self.len = forces.shape[0]

            # Load all data
            self.data = {v: npz_file[v] for v in props.values()}

            # Verify consistency
            positions = self.data[props[Props.positions]]
            if self.is_ragged:
                assert positions.dtype == np.dtype("O"), "Positions must also be object array if forces are"
                # Verify each configuration has matching number of atoms
                for pos, force in zip(positions, forces, strict=True):
                    assert len(pos) == len(force), f"Mismatch in atoms: {len(pos)} positions vs {len(force)} forces"
            else:
                assert positions.shape[0] == self.len, "Inconsistent number of configurations"
                assert positions.shape[1] == forces.shape[1], "Inconsistent number of atoms"
                assert positions.shape[2] == forces.shape[2] == 3, "Positions and forces must be 3D"

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> dict:
        sample = {
            self.props[Props.energy]: self.data[self.props[Props.energy]][idx],
            self.props[Props.forces]: convert_force(
                self.data[self.props[Props.forces]][idx], from_unit=self.force_unit, to_unit="Hartree/Bohr"
            ),
            self.props[Props.positions]: convert_coordinates(
                self.data[self.props[Props.positions]][idx], from_unit=self.coord_unit, to_unit="Bohr"
            ),
            self.props[Props.atomic_numbers]: (
                self.data[self.props[Props.atomic_numbers]][idx]
                if self.is_ragged
                else self.data[self.props[Props.atomic_numbers]]
            ),
        }

        if Props.charge in self.props:
            sample[self.props[Props.charge]] = self.data[self.props[Props.charge]][idx]

        return sample

    @property
    def is_variable_size(self) -> bool:
        """Returns whether the dataset contains variable-sized molecules."""
        return self.is_ragged

    def get_size_range(self) -> tuple[int, int]:
        """Returns the (min, max) number of atoms across all molecules in the dataset."""
        if not self.is_ragged:
            n_atoms = self.data[self.props[Props.positions]][0].shape[0]
            return n_atoms, n_atoms

        positions = self.data[self.props[Props.positions]]
        sizes = [len(pos) for pos in positions]
        return min(sizes), max(sizes)
