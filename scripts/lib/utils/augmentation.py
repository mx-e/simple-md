import torch as th
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger


def generate_equidistant_rotations(N, device="cpu", optimize: bool = True) -> th.Tensor:
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    indices = th.arange(N, dtype=th.float32, device=device)

    theta = 2 * np.pi * indices / phi
    z = 1 - (2 * indices + 1) / N
    radius = th.sqrt(1 - z * z)

    x = radius * th.cos(theta)
    y = radius * th.sin(theta)
    z = z

    points = th.stack([x, y, z], dim=1)
    points = points / th.norm(points, dim=1, keepdim=True)
    if optimize:
        points = optimize_rotation_distribution(points)

    R = th.stack([th.tensor(vector_to_rot_matrix(points[i].numpy())) for i in range(N)], dim=0)
    return R


def visualize_rotations(rotation_matrices: th.Tensor, save_path: str = "rotation_visualization.png") -> None:
    R = rotation_matrices.numpy()
    N = len(R)
    points = np.array([rot_matrix_to_vector(R[i]) for i in range(N)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    front_mask = points[:, 2] >= 0
    front_points = points[front_mask]
    ax1.plot(circle_x, circle_y, "k-", alpha=0.2)
    ax1.scatter(
        front_points[:, 0], front_points[:, 1], c="red", alpha=0.6, label=f"Front points ({np.sum(front_mask)})"
    )
    ax1.set_title("Front Hemisphere (z ≥ 0)")

    back_mask = points[:, 2] < 0
    back_points = points[back_mask]
    ax2.plot(circle_x, circle_y, "k-", alpha=0.2)
    ax2.scatter(back_points[:, 0], back_points[:, 1], c="blue", alpha=0.6, label=f"Back points ({np.sum(back_mask)})")
    ax2.set_title("Back Hemisphere (z < 0)")

    for ax in [ax1, ax2]:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def angle_between_rotations(R1: th.Tensor, R2: th.Tensor) -> float:
    R_relative = R1 @ R2.T
    trace = np.trace(R_relative)
    trace = np.clip(trace, -3.0, 3.0)
    theta = np.arccos((trace - 1) / 2)
    return theta  # Returns angle in radians


def analyze_rotation_distribution(rotation_matrices: th.Tensor) -> dict:
    N = len(rotation_matrices)
    angles = []
    for i in range(N):
        for j in range(i + 1, N):
            angle_radians = angle_between_rotations(rotation_matrices[i], rotation_matrices[j])
            angle_degrees = np.degrees(angle_radians)
            angles.append(angle_degrees)

    angles = np.array(angles)
    stats = {
        "min_angle": np.min(angles),
        "max_angle": np.max(angles),
        "mean_angle": np.mean(angles),
        "std_angle": np.std(angles),
    }
    return stats


def optimize_rotation_distribution(points: th.Tensor, num_steps: int = 1000, lr: float = 0.001) -> th.Tensor:
    def objective_func(points: th.Tensor) -> th.Tensor:
        dists = th.norm(points[:, None] - points[None], dim=-1)  # (N, N)
        mask = th.eye(len(points), device=points.device) == 1
        dists = dists[~mask]
        return -dists.sum()

    logger.info("Optimizing rotation distribution...")
    logger.info(f"Initial loss = {objective_func(points).item()}")
    points = points.clone().detach().requires_grad_(True)
    optimizer = th.optim.Adam([points], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        norm_points = points / th.norm(points, dim=1, keepdim=True)
        loss = objective_func(norm_points)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            logger.info(f"Step {step}: Loss = {loss.item()}")
    points = points / th.norm(points, dim=1, keepdim=True)

    logger.info(f"Final loss = {objective_func(points).item()}")
    return points.detach().clone()


def vector_to_rot_matrix(target_vec: np.ndarray) -> np.ndarray:
    target = np.asarray(target_vec, dtype=np.float64)
    target = target / np.linalg.norm(target)

    ref = np.array([0, 0, 1])

    axis = np.cross(ref, target)

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        if np.dot(ref, target) > 0:
            return np.eye(3)
        else:
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    axis = axis / axis_norm

    cos_angle = np.dot(ref, target)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * (K @ K)
    return R


def rot_matrix_to_vector(R: np.ndarray) -> np.ndarray:
    ref_vector = np.array([0, 0, 1])
    target_vector = R @ ref_vector
    return target_vector
