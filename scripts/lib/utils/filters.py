from typing import Tuple

import torch as th


def compute_net_torque(
    positions: th.Tensor,
    forces: th.Tensor,
    n_nodes: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute the net torque on a system of particles."""
    # TODO: check why this does not actually compute the center of mass but rather the molecule centroid, try removing this entire part, since positions should already be normalized to the center of mass
    b, n, _ = positions.shape
    com = positions.mean(dim=1)  # (b, 3)
    com_repeat = th.repeat_interleave(com, n_nodes, dim=0)  # (b, n, 3)
    flat_positions = positions.flatten(0, 1)  # (b * n, 3)
    com_relative_positions = flat_positions - com_repeat  # (b * n, 3)
    flat_forces = forces.flatten(0, 1)  # (b * n, 3)
    torques = th.linalg.cross(com_relative_positions, flat_forces)  # (b * n, 3)
    net_torque = torques.view(b, n, 3).sum(dim=1)  # (b, 3)
    return net_torque, com_relative_positions


def remove_net_torque(
    positions: th.Tensor,
    forces: th.Tensor,
    n_nodes: th.Tensor,
) -> th.Tensor:
    assert th.all(n_nodes[0] == n_nodes), (
        "This function only works for batches of graphs with the same number of nodes"
    )
    b, n, _ = positions.shape
    tau_total, r = compute_net_torque(positions, forces, n_nodes)  # (b, 3), (b * n, 3)

    # Compute scalar s per graph: sum_i ||r_i||^2
    r_squared = th.sum(r**2, dim=1)  # (b * n)
    s = r_squared.view(b, n).sum(dim=1)  # (b)

    # Compute matrix S per graph: sum_i outer(r_i, r_i)
    r_unsqueezed = r.unsqueeze(2)  # (b*n, 3, 1)
    r_T_unsqueezed = r.unsqueeze(1)  # (b*n, 1, 3)
    outer_products = r_unsqueezed @ r_T_unsqueezed  # (b*n, 3, 3)
    S = outer_products.view(b, n, 3, 3).sum(dim=1)  # (b, 3, 3)

    # Compute M = S - sI
    I = th.eye(3, device=positions.device).unsqueeze(0).expand(b, -1, -1)  # (b, 3, 3)
    M = S - (s.view(-1, 1, 1)) * I  # (b, 3, 3)

    # Right-hand side vector b per graph
    vb = -tau_total  # (b, 3)

    # Solve M * mu = b for mu per graph
    try:
        mu = th.linalg.solve(M, vb.unsqueeze(2)).squeeze(2)  # (b, 3)
    except RuntimeError:
        # Handle singular matrix M by using the pseudo-inverse
        M_pinv = th.linalg.pinv(M)  # Shape: (B, 3, 3)
        mu = th.bmm(M_pinv, vb.unsqueeze(2)).squeeze(2)  # (b, 3)

    # Compute adjustments to forces
    mu_batch = th.repeat_interleave(mu, n_nodes, dim=0)  # (b*n, 3)
    forces_delta = th.linalg.cross(r, mu_batch)  # (b*n, 3)
    forces_delta = forces_delta.view(b, n, 3)  # (b, n, 3)

    # Adjusted forces
    adjusted_forces = forces + forces_delta  # (b, n, 3)

    return adjusted_forces


def remove_net_force(
    forces: th.Tensor,
    n_nodes: th.Tensor,
) -> th.Tensor:
    assert th.all(n_nodes[0] == n_nodes), (
        "This function only works for batches of graphs with the same number of nodes"
    )
    b, n, _ = forces.shape
    net_force = forces.sum(dim=1)  # (b, 3)
    num_nodes = n_nodes.unsqueeze(1)  # (b, 1)
    force_per_node = net_force / num_nodes  # (b, 3)
    forces_per_node_repeated = force_per_node.repeat_interleave(n_nodes, dim=0).view(b, n, 3)  # (b, n, 3)
    assert forces_per_node_repeated.shape == forces.shape, "Shapes do not match"
    adjusted_forces = forces - forces_per_node_repeated  # (b, n, 3)
    return adjusted_forces
