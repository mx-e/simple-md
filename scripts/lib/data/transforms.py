from enum import Enum
from functools import wraps

import torch as th
from ase import Atoms
from ase.data import atomic_masses
from ase.neighborlist import neighbor_list
from lib.types import Property as Props
from lib.types import property_dtype
from lib.utils.augmentation import get_random_reflections, get_random_rotations
from loguru import logger


class Transform(Enum):
    center_on_centroid = "center_on_centroid"
    compute_neigbourhoods = "compute_neigbourhoods"
    dynamic_batch_size = "dynamic_batch_size"
    augment_positions = "augment_positions"
    add_default_multiplicity = "add_default_multiplicity"
    add_default_charge = "add_default_charge"


def _apply_molwise_func(batch, func, new_props, **kwargs) -> tuple[dict, list]:
    for _, sample in enumerate(batch):
        sample.update(func(sample, **kwargs))
    return batch, new_props


def apply_molwise(new_props) -> callable:
    def create_wrapper(func) -> callable:
        @wraps(func)
        def wrapper(batch, **kwargs) -> tuple[dict, list]:
            return _apply_molwise_func(batch, func, new_props, **kwargs)

        # needed for multiprocessing
        wrapper.__module__ = __name__
        return wrapper

    # needed for multiprocessing
    create_wrapper.__module__ = __name__
    return create_wrapper


@apply_molwise(new_props=[])
def center_positions_on_centroid(mol) -> dict:
    positions = mol[Props.positions]  # (n_atoms, 3)
    centroid = (positions).mean(dim=0, keepdim=True)
    new_positions = positions - centroid
    return {Props.positions: new_positions}


@apply_molwise(new_props=[Props.charge])
def add_default_charge(_) -> dict:
    return {Props.charge: th.tensor([0], dtype=property_dtype[Props.charge])}


@apply_molwise(new_props=[Props.multiplicity])
def add_default_multiplicity(_) -> dict:
    return {Props.multiplicity: th.tensor([1], dtype=property_dtype[Props.multiplicity])}


@apply_molwise(new_props=[])
def center_positions_on_center_of_mass(mol) -> dict:
    masses = th.tensor(atomic_masses[mol[Props.atomic_numbers]], dtype=th.float32)
    com = (masses.unsqueeze(-1) * mol[Props.positions]).sum(0) / masses.sum()
    return {Props.positions: mol[Props.positions] - com}


@apply_molwise(
    new_props=[Props.i_idx, Props.j_idx],
)
def compute_neigbourhoods(mol, cutoff) -> dict:
    positions = mol[Props.positions]  # (n_atoms, 3)
    atomic_nums = mol[Props.atomic_numbers]  # (n_atoms,)
    at = Atoms(atomic_nums, positions, pbc=False)
    idx_i, idx_j = neighbor_list("ij", at, cutoff)
    return {
        Props.i_idx: th.tensor(idx_i, dtype=property_dtype[Props.i_idx]),
        Props.j_idx: th.tensor(idx_j, dtype=property_dtype[Props.j_idx]),
    }


def dynamic_batch_size(batch, cutoff=30) -> dict:
    batch_size, n_atoms = batch[Props.mask].shape
    cost = n_atoms**3 / cutoff**3
    if cost > 1:
        cutoff_factor = 1 / cost
        new_batch_size = int(batch_size * cutoff_factor)
        logger.info(f"Reducing batch size from {batch_size} to {new_batch_size}")
        for k, v in batch.items():
            batch[k] = v[:new_batch_size]

    return batch


# post batch preprocessor
def augment_positions(
    batch,
    augmentation_mult=1,
    random_rotation=False,
    random_reflection=False,
) -> dict:
    if augmentation_mult <= 1:
        return batch
    for k, v in batch.items():
        batch[k] = v.repeat_interleave(augmentation_mult, dim=0)

    positions = batch[Props.positions]  # (n_batches, n_atoms, 3)
    forces = batch[Props.forces]  # (n_batches, n_atoms, 3)

    n_batches, _, _ = positions.size()
    if random_rotation:
        R = get_random_rotations(n_batches, positions.device)
        positions = th.bmm(positions, R)
        forces = th.bmm(forces, R)

    if random_reflection:
        H = get_random_reflections(n_batches, positions.device, reflection_share=0.5)
        positions = th.bmm(positions, H)
        forces = th.bmm(forces, H)

    if Props.dipole in batch:
        dipole = batch[Props.dipole]
        dipole = th.bmm(dipole.unsqueeze(1), R).squeeze(1)
        dipole = th.bmm(dipole.unsqueeze(1), H).squeeze(1)
        batch[Props.dipole] = dipole

    batch[Props.positions] = positions
    batch[Props.forces] = forces
    return batch


pre_collate_transforms = {
    Transform.center_on_centroid: center_positions_on_centroid,
    Transform.compute_neigbourhoods: compute_neigbourhoods,
    Transform.add_default_charge: add_default_charge,
    Transform.add_default_multiplicity: add_default_multiplicity,
}

post_collate_transforms = {
    Transform.dynamic_batch_size: dynamic_batch_size,
    Transform.augment_positions: augment_positions,
}
