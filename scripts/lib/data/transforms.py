from enum import Enum
from functools import wraps

import torch as th
from ase import Atoms
from ase.data import atomic_masses
from ase.neighborlist import neighbor_list
from lib.types import Property as Props, Props_dtype
from loguru import logger


class Transform(Enum):
    center_on_centroid = "center_on_centroid"
    compute_neigbourhoods = "compute_neigbourhoods"
    dynamic_batch_size = "dynamic_batch_size"
    augment_positions = "augment_positions"


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


@apply_molwise(new_props=[])
def center_positions_on_center_of_mass(mol) -> dict:
    masses = th.tensor(atomic_masses[mol[Props.atomic_numbers]], dtype=th.float32)
    com = (masses.unsqueeze(-1) * mol[Props.position]).sum(0) / masses.sum()
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
        Props.i_idx: th.tensor(idx_i, dtype=Props_dtype[Props.i_idx]),
        Props.j_idx: th.tensor(idx_j, dtype=Props_dtype[Props.j_idx]),
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


def get_random_rotations(n_samples, device) -> th.Tensor:
    # Generate random points on the surface of a 4D hypersphere
    u1, u2, u3 = th.rand(n_samples, 3).chunk(3, dim=1)  # (n_samples, 1)

    # Convert to quaternion
    a = th.sqrt(1 - u1) * th.sin(2 * th.pi * u2)
    b = th.sqrt(1 - u1) * th.cos(2 * th.pi * u2)
    c = th.sqrt(u1) * th.sin(2 * th.pi * u3)
    d = th.sqrt(u1) * th.cos(2 * th.pi * u3)

    # Convert quaternion to rotation matrix
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    ab, ac, ad = a * b, a * c, a * d
    bc, bd, cd = b * c, b * d, c * d

    R = th.stack(
        [
            aa + bb - cc - dd,
            2 * (bc - ad),
            2 * (bd + ac),
            2 * (bc + ad),
            aa - bb + cc - dd,
            2 * (cd - ab),
            2 * (bd - ac),
            2 * (cd + ab),
            aa - bb - cc + dd,
        ],
        dim=-1,
    ).reshape(n_samples, 3, 3)  # (n_samples, 3, 3)

    return R.to(device)


def get_random_reflections(n_samples, device, reflection_share=0.5, eps=1e-9) -> th.Tensor:
    # get random normal vectors
    normals = th.randn(n_samples, 3).to(device)  # (n_samples, 3)
    normals = normals / (th.norm(normals, dim=1, keepdim=True) + eps)  # (n_samples, 3)

    # get householder matrix
    normals = normals.unsqueeze(2)
    outer = th.matmul(normals, normals.transpose(1, 2))  # (n_samples, 3, 3)
    identity = th.eye(3, dtype=normals.dtype, device=normals.device).unsqueeze(0)  # (1, 3, 3)
    householder = identity.repeat(n_samples, 1, 1)  # (n_samples, 3, 3)
    # selectively reflect
    sample_mask = th.rand(n_samples) < reflection_share
    householder[sample_mask] -= 2 * outer[sample_mask]
    return householder


pre_collate_transforms = {
    Transform.center_on_centroid: center_positions_on_centroid,
    Transform.compute_neigbourhoods: compute_neigbourhoods,
}

post_collate_transforms = {
    Transform.dynamic_batch_size: dynamic_batch_size,
    Transform.augment_positions: augment_positions,
}
