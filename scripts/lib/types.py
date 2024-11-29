from enum import Enum
from dataclasses import dataclass, field
from typing import Literal, Callable

from frozendict import frozendict
import torch as th
from torch.utils.data import Dataset

NODE_FEATURES_OFFSET = 128


class Property(Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def _missing_(cls, value):
        # This allows Property["energy"] to work the same as Property.energy
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")

    energy = "energy"
    formation_energy = "formation_energy"
    charge = "charge"
    multiplicity = "multiplicity"
    atomic_numbers = "atomic_numbers"
    mask = "mask"
    forces = "forces"
    positions = "positions"
    dipole = "dipole"
    mol_idx = "mol_idx"
    i_idx = "i_idx"
    j_idx = "j_idx"
    energy_atomref = "energy_atomref"
    formation_energy_atomref = "formation_energy_atomref"
    r_ij = "r_ij"
    i_idx_local = "i_idx_local"
    j_idx_local = "j_idx_local"
    _meta = "_meta"


class PropertyType(Enum):
    mol_wise = 1
    atom_wise = 2
    edge_wise = 3


property_dims = frozendict(
    {
        Property.energy: 1,
        Property.formation_energy: 1,
        Property.energy_atomref: 1,
        Property.charge: 1,
        Property.multiplicity: 1,
        Property.atomic_numbers: 1,
        Property.mask: 1,
        Property.forces: 3,
        Property.positions: 3,
        Property.mol_idx: 1,
        Property.dipole: 3,
        Property.i_idx: 1,
        Property.j_idx: 1,
        Property.energy_atomref: 1,
        Property.formation_energy_atomref: 1,
        Property.r_ij: 3,
        Property.i_idx_local: 1,
        Property.j_idx_local: 1,
    }
)

property_dtype = frozendict(
    {
        Property.energy: th.float32,
        Property.formation_energy: th.float32,
        Property.charge: th.int64,
        Property.multiplicity: th.int64,
        Property.atomic_numbers: th.int64,
        Property.mask: th.bool,
        Property.forces: th.float32,
        Property.positions: th.float32,
        Property.mol_idx: th.int64,
        Property.dipole: th.float32,
        Property.i_idx: th.int64,
        Property.j_idx: th.int64,
        Property.energy_atomref: th.float32,
        Property.formation_energy_atomref: th.float32,
        Property.r_ij: th.float32,
        Property.i_idx_local: th.int64,
        Property.j_idx_local: th.int,
    }
)

property_type = frozendict(
    {
        Property.energy: PropertyType.mol_wise,
        Property.formation_energy: PropertyType.mol_wise,
        Property.charge: PropertyType.mol_wise,
        Property.multiplicity: PropertyType.mol_wise,
        Property.atomic_numbers: PropertyType.atom_wise,
        Property.mask: PropertyType.atom_wise,
        Property.forces: PropertyType.atom_wise,
        Property.positions: PropertyType.atom_wise,
        Property.mol_idx: PropertyType.edge_wise,
        Property.dipole: PropertyType.mol_wise,
        Property.i_idx: PropertyType.edge_wise,
        Property.j_idx: PropertyType.edge_wise,
        Property.energy_atomref: PropertyType.mol_wise,
        Property.formation_energy_atomref: PropertyType.mol_wise,
        Property.r_ij: PropertyType.atom_wise,
        Property.energy_atomref: PropertyType.mol_wise,
        Property.i_idx_local: PropertyType.edge_wise,
        Property.j_idx_local: PropertyType.edge_wise,
    }
)

edge_props_to_local_map = frozendict(
    {Property.i_idx: Property.i_idx_local, Property.j_idx: Property.j_idx_local}
)


@dataclass
class FeatureStats:
    mean: float
    var: float
    atomref: dict[int, float]


DatasetStats = dict[Property, FeatureStats]


class Split(Enum):
    train = "train"
    val = "val"
    test = "test"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def _missing_(cls, value):
        # This allows Property["energy"] to work the same as Property.energy
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


@dataclass
class DatasetSplits:
    splits: dict[Split, Dataset]
    dataset_props: dict[Property, str]
    dataset_stats: DatasetStats | None = None


@dataclass
class PipelineConfig:
    pre_collate_processors: list[Callable[[list[dict]], list[dict]]] = field(
        default_factory=list
    )
    pre_collate_processors_val: list[Callable[[list[dict]], list[dict]]] | None = None
    post_collate_processors: list[Callable[[dict], dict]] = field(default_factory=list)
    post_collate_processors_val: list[Callable[[dict], dict]] | None = None
    collate_type: Literal["tall", "flat"] = "tall"
    batch_size_impact: float = 1.0
    needed_props: list[Property] = field(default_factory=list)

    def __post_init__(self):
        if self.pre_collate_processors_val is None:
            self.pre_collate_processors_val = self.pre_collate_processors
        if self.post_collate_processors_val is None:
            self.post_collate_processors_val = self.post_collate_processors
