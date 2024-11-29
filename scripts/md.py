#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data /home/maxi/MOLECULAR_ML/5_refactored_repo/container.sif python
from functools import partial
from pathlib import Path
import csv
from typing import Literal

import torch as th
import torch.nn as nn
import numpy as np
from omegaconf import MISSING
import torch.multiprocessing as mp
from hydra_zen import load_from_yaml, instantiate

from matplotlib import pyplot as plt
from loguru import logger
from ase import units
from ase.io import read, write, trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin  # Added Langevin thermostat
from ase.md.npt import NPT  # Added NPT with Nosé-Hoover
from ase.md.nvtberendsen import NVTBerendsen  # Added Berendsen thermostat
from ase.calculators.calculator import Calculator, all_changes

from conf.base_conf import configure_main, BaseConfig
from lib.utils.dist import setup_device
from lib.utils.checkpoint import load_checkpoint
from lib.types import Property as Props
from lib.data.loaders import collate_fn, batch_tall
from lib.data.transforms import center_positions_on_centroid, get_random_rotations
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run


BOHR_TO_ANG = 0.529177249  # Bohr to Angstrom
HARTREE_TO_EV = 27.211386245988  # Hartree to eV
ANG_TO_BOHR = 1.0 / 0.529177249
FORCE_CONVERSION = HARTREE_TO_EV / BOHR_TO_ANG


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,
    temperature: float = 300,
    timestep: float = 0.5,
    n_data_aug: int = 1,
    step_wise_random: bool = False,
    n_steps: int = 1000,
    thermostat_type: Literal[
        "nose_hoover", "berendsen", "andersen", "langevin"
    ] = "langevin",
    thermostat_taut: float = 100.0,
    init_struct_dir: Path = "data_md",
    init_struct: Literal[
        "15_ala",
        "buckyball_catcher",
        "c60",
        "dichlormethan",
        "ethanol",
        "hydrogen",
        "silver_trimer",
    ] = "ethanol",
    model_run_dir: Path = MISSING,
    checkpoint_name: str = "best_model",
):
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    job_dir = get_hydra_output_dir()
    ctx, device = setup_device()
    model_run_conf_path = model_run_dir / ".hydra" / "config.yaml"
    model_run_conf = load_from_yaml(model_run_conf_path)
    model_conf = model_run_conf["train"]["model"]
    model = instantiate(model_conf)
    checkpoint_path = Path(model_run_dir) / "ckpts" / (checkpoint_name + ".pth")
    load_checkpoint(model, checkpoint_path)
    model.eval().to(device)

    # prepare results directory, load initial structure
    results_dir = Path(job_dir) / "md_results"
    results_dir.mkdir(exist_ok=True)
    init_struct_path = Path(init_struct_dir) / (init_struct + ".xyz")
    atoms = read(init_struct_path)

    if thermostat_type.lower() == "nose_hoover":
        atoms.set_cell(
            [
                [120.0, 0.0, 0.0],  # Gives ~25Å buffer on each side
                [0.0, 120.0, 0.0],
                [0.0, 0.0, 120.0],
            ]
        )
        atoms.set_pbc(False)

    if n_data_aug > 1:
        logger.info(f"Generating {n_data_aug} equidistant rotations...")
        rotations = generate_equidistant_rotations(n_data_aug)
        # Create visualization and save to file
        rotations_path = Path(job_dir) / "data_aug" / "rotation_visualization.png"
        rotations_path.parent.mkdir(exist_ok=True)
        visualize_rotations(rotations, rotations_path)
        stats = analyze_rotation_distribution(rotations)
        logger.info(f"Rotation Distribution Statistics: {stats}")

    # Run MD simulation
    traj_path = Path(job_dir) / "md_results" / "md_trajectory.traj"
    energy_tracker = run_md_simulation(
        atoms=atoms,
        model=model,
        ctx=ctx,
        device=device,
        temperature=temperature,
        timestep=timestep,
        rotations=rotations.to(device) if n_data_aug > 1 else None,
        step_wise_random_aug=step_wise_random,
        steps=n_steps,
        trajectory_file=traj_path,
        thermostat_type=thermostat_type,
        taut=thermostat_taut,
    )
    # Save and plot results
    energy_tracker.save_data(results_dir)
    energy_tracker.plot(results_dir / "md_analysis.png")

    # Print final statistics
    final_stats = energy_tracker.get_stats()
    logger.info("Simulation Statistics:")
    logger.info(
        f"  Average Temperature: {final_stats['avg_temperature']:.1f} ± {final_stats['temp_std']:.1f} K"
    )
    logger.info(
        f"  Average Kinetic Energy: {final_stats['avg_kinetic']:.3f} ± {final_stats['kinetic_std']:.3f} eV"
    )


def run_md_simulation(
    atoms,
    model,
    ctx,
    device,
    temperature=300,  # K
    timestep=0.5,  # fs
    rotations=None,
    step_wise_random_aug=False,
    steps=1000,
    trajectory_file="md_trajectory.traj",
    thermostat_type="Nose-Hoover",  # Added thermostat selection
    taut=100.0,  # Thermostat time constant in fs
):
    """Run MD simulation using the trained model with thermostat control"""

    # Set up calculator
    calculator = MLCalculator(
        model,
        device,
        ctx,
        rotations=rotations,
        step_wise_random_aug=step_wise_random_aug,
    )
    atoms.calc = calculator

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # Set up dynamics with selected thermostat
    if thermostat_type.lower() == "nose_hoover":
        # Nosé-Hoover thermostat (NPT ensemble)
        # The time constant is converted to ASE units
        taut_ase = taut * units.fs
        dyn = NPT(
            atoms,
            timestep * units.fs,
            temperature_K=temperature,
            externalstress=0.0,
            ttime=taut_ase,
            pfactor=None,
        )
    elif thermostat_type.lower() == "berendsen":
        # Berendsen thermostat (NVT ensemble)
        dyn = NVTBerendsen(
            atoms, timestep * units.fs, temperature_K=temperature, taut=taut * units.fs
        )
    elif thermostat_type.lower() == "andersen":
        # Andersen thermostat
        dyn = Andersen(
            atoms, timestep * units.fs, temperature_K=temperature, andersen_prob=0.01
        )
    elif thermostat_type.lower() == "langevin":
        # Langevin thermostat
        # friction parameter is 1/taut
        dyn = Langevin(
            atoms,
            timestep * units.fs,
            temperature_K=temperature,
            friction=1.0 / (taut * units.fs),
        )
    else:
        raise ValueError(f"Unknown thermostat type: {thermostat_type}")

    # Set up trajectory file and energy tracker
    traj = trajectory.Trajectory(trajectory_file, "w", atoms)
    energy_tracker = MDEnergyTracker(atoms, timestep, temperature)
    xyz_trajectory = []

    def save_frame():
        frame = atoms.copy()
        frame.info["comment"] = (
            f"time={len(xyz_trajectory) * timestep * 10:.1f}fs "
            f"temp={atoms.get_temperature():.1f}K "
            f"thermostat={thermostat_type}"
        )
        xyz_trajectory.append(frame)

    # Attach observers
    dyn.attach(traj.write, interval=10)
    dyn.attach(save_frame, interval=10)
    dyn.attach(energy_tracker, interval=10)

    # Run dynamics with improved monitoring
    logger.info(
        f"Starting MD simulation with {thermostat_type} thermostat for {steps} steps..."
    )
    logger.info(f"Target temperature: {temperature}K")
    logger.info(f"Thermostat time constant: {taut}fs")

    for i in range(steps):
        dyn.run(1)
        if i % 100 == 0:
            temp = atoms.get_temperature()
            temp_diff = abs(temp - temperature)
            temp_deviation_percent = (temp_diff / temperature) * 100

            logger.info(f"Step {i}:")
            logger.info(
                f"  Temperature: {temp:.1f}K (Target: {temperature}K, "
                f"Deviation: {temp_deviation_percent:.1f}%)"
            )
            logger.info(f"  Kinetic Energy: {atoms.get_kinetic_energy():.3f} eV")
            # Check simulation stability
            if np.any(np.isnan(atoms.get_positions())) or np.any(
                np.isnan(atoms.get_velocities())
            ):
                logger.error(
                    "Simulation unstable: NaN detected in positions or velocities!"
                )
                break

            # Check molecular integrity
            distances = atoms.get_all_distances()
            max_dist = np.max(distances)
            if max_dist > 10.0:  # Å
                logger.warning(
                    f"Simulation unstable: Atoms too far apart ({max_dist:.2f} Å)!"
                )

    logger.info("MD simulation completed!")

    xyz_file = str(Path(trajectory_file).with_suffix(".xyz"))
    # remove cell information
    for frame in xyz_trajectory:
        frame.set_cell([0, 0, 0])
        frame.set_pbc(False)
    write(xyz_file, xyz_trajectory)

    logger.info(f"Trajectory saved to {trajectory_file} and {xyz_file}")

    return energy_tracker


class MLCalculator(Calculator):
    """Calculator interfacing the trained ML model with ASE"""

    implemented_properties = ["forces"]
    not_implemented_properties = ["energy"]

    def __init__(self, model, device, ctx, rotations=None, step_wise_random_aug=False):
        super().__init__()
        self.model = model
        self.device = device
        self.ctx = ctx
        self.model.eval()
        pre_batch_preprocessors = [center_positions_on_centroid]
        props = {
            Props.positions: "positions",
            Props.atomic_numbers: "atomic_numbers",
            Props.charge: "charge",
            Props.multiplicity: "multiplicity",
        }
        self.collate_fn = partial(
            collate_fn,
            device=device,
            batch_func=batch_tall,
            pre_batch_preprocessors=pre_batch_preprocessors,
            props=props,
        )
        self.rotations = rotations
        self.step_wise_random_aug = step_wise_random_aug

    def convert_atoms_to_model_input(self, atoms, rotations):
        """Convert ASE Atoms object to model input format"""
        positions_bohr = atoms.positions * ANG_TO_BOHR
        spin = atoms.get_initial_magnetic_moments().sum()
        multiplicity = 2 * abs(spin) + 1
        data = {
            "positions": positions_bohr,
            "atomic_numbers": atoms.numbers,
            "charge": 1,  # atoms.get_initial_charges().sum(),
            "multiplicity": multiplicity,
        }

        data = self.collate_fn([data])

        reverse_rotations = None
        if rotations is not None:
            for k, v in data.items():
                expand_shape = (len(rotations),) + tuple(-1 for _ in range(v.dim() - 1))
                data[k] = v.expand(expand_shape)
            if self.step_wise_random_aug:
                random_offset_rotation = get_random_rotations(1, self.device).expand(
                    len(rotations), -1, -1
                )
                rotations_step = th.bmm(rotations, random_offset_rotation)
            else:
                rotations_step = rotations

            data[Props.positions] = th.bmm(
                data[Props.positions], rotations_step
            ).squeeze()
            reverse_rotations = rotations_step.transpose(1, 2)
        return data, reverse_rotations

    @th.no_grad()
    def calculate(self, atoms=None, properties=["forces"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert atoms to model input
        data, reverse_rotations = self.convert_atoms_to_model_input(
            atoms, self.rotations
        )

        # Get predictions from model
        with self.ctx:
            outputs = self.model(data)

        if self.rotations is not None:
            outputs[Props.forces] = th.bmm(outputs[Props.forces], reverse_rotations)
            mean_force_predictions = outputs[Props.forces].mean(dim=0)
        else:
            mean_force_predictions = outputs[Props.forces].squeeze()
        mean_force_predictions *= FORCE_CONVERSION
        # Convert outputs to calculator format
        self.results = {
            "forces": mean_force_predictions.squeeze().cpu().numpy(),
        }


def visualize_rotations(
    rotation_matrices: th.Tensor, save_path: str = "rotation_visualization.png"
):
    """
    Visualize rotation matrices by showing where they map the standard basis vectors.
    Creates both 3D and 2D projections of the rotations.

    Args:
        rotation_matrices: (N, 3, 3) tensor of rotation matrices
        save_path: path to save the visualization
    """
    # Convert to numpy for matplotlib
    R = rotation_matrices.numpy()
    N = len(R)

    # Create standard basis vectors
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    # Calculate rotated versions of each basis vector
    rotated_e1 = np.array([R[i] @ e1 for i in range(N)])
    rotated_e2 = np.array([R[i] @ e2 for i in range(N)])
    rotated_e3 = np.array([R[i] @ e3 for i in range(N)])

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))

    # 3D plot
    ax1 = fig.add_subplot(121, projection="3d")

    # Plot rotated basis vectors
    ax1.scatter(
        rotated_e1[:, 0],
        rotated_e1[:, 1],
        rotated_e1[:, 2],
        c="r",
        label="x-axis",
        alpha=0.6,
    )
    ax1.scatter(
        rotated_e2[:, 0],
        rotated_e2[:, 1],
        rotated_e2[:, 2],
        c="g",
        label="y-axis",
        alpha=0.6,
    )
    ax1.scatter(
        rotated_e3[:, 0],
        rotated_e3[:, 1],
        rotated_e3[:, 2],
        c="b",
        label="z-axis",
        alpha=0.6,
    )

    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    # Set labels and title
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Distribution of {N} Rotations\n(Rotated Basis Vectors)")
    ax1.legend()

    # Create 2D projections subplot
    ax2 = fig.add_subplot(122)

    # Plot XY projection
    ax2.scatter(rotated_e1[:, 0], rotated_e1[:, 1], c="r", alpha=0.3, label="x-axis")
    ax2.scatter(rotated_e2[:, 0], rotated_e2[:, 1], c="g", alpha=0.3, label="y-axis")
    ax2.scatter(rotated_e3[:, 0], rotated_e3[:, 1], c="b", alpha=0.3, label="z-axis")

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--", alpha=0.3)
    ax2.add_artist(circle)

    # Set labels and title
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("XY Projection of Rotations")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_rotation_distribution(rotation_matrices: th.Tensor):
    """
    Analyze the distribution of rotations by computing pairwise angles.

    Args:
        rotation_matrices: (N, 3, 3) tensor of rotation matrices

    Returns:
        dict: Statistics about the rotation distribution
    """
    N = len(rotation_matrices)

    # Compute pairwise angles between rotations
    angles = []
    for i in range(N):
        for j in range(i + 1, N):
            # Compute relative rotation
            R_rel = th.mm(rotation_matrices[i], rotation_matrices[j].t())

            # Convert to angle (in degrees)
            theta = (
                th.acos(th.clamp((th.trace(R_rel) - 1) / 2, -1.0, 1.0)) * 180 / np.pi
            )
            angles.append(theta.item())

    angles = np.array(angles)
    stats = {
        "min_angle": np.min(angles),
        "max_angle": np.max(angles),
        "mean_angle": np.mean(angles),
        "std_angle": np.std(angles),
    }
    return stats


def generate_equidistant_rotations(N, device="cpu"):
    """
    Generate N approximately equidistant rotation matrices in SO(3).
    Uses the Fibonacci sphere method for point distribution, combined with
    quaternion representation for smooth rotation interpolation.

    Args:
        N (int): Number of rotation matrices to generate
        device (str): Pythdevice to use ('cpu' or 'cuda')

    Returns:
        th.Tensor: (N, 3, 3) tensor containing N rotation matrices
    """
    # Generate points on a Fibonacci sphere
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    indices = th.arange(N, dtype=th.float32, device=device)

    # Calculate spherical coordinates
    theta = 2 * np.pi * indices / phi
    z = 1 - (2 * indices + 1) / N
    radius = th.sqrt(1 - z * z)

    # Convert to Cartesian coordinates
    x = radius * th.cos(theta)
    y = radius * th.sin(theta)
    z = z

    # Stack into points
    points = th.stack([x, y, z], dim=1)
    points = points / th.norm(points, dim=1, keepdim=True)

    # Convert points to rotation matrices using quaternions
    def points_to_quaternions(points):
        """Convert points on unit sphere to quaternions."""
        # Use the method described in "Uniform Random Rotations" by Ken Shoemake
        u = th.rand(N, dtype=th.float32, device=device)
        v = th.rand(N, dtype=th.float32, device=device)
        w = th.rand(N, dtype=th.float32, device=device)

        # Convert uniform random numbers to quaternion
        q1 = th.sqrt(1 - u) * th.sin(2 * np.pi * v)
        q2 = th.sqrt(1 - u) * th.cos(2 * np.pi * v)
        q3 = th.sqrt(u) * th.sin(2 * np.pi * w)
        q4 = th.sqrt(u) * th.cos(2 * np.pi * w)

        return th.stack([q1, q2, q3, q4], dim=1)

    def quaternion_to_rotation_matrix(quaternion):
        """Convert quaternions to rotation matrices."""
        q0, q1, q2, q3 = (
            quaternion[:, 0],
            quaternion[:, 1],
            quaternion[:, 2],
            quaternion[:, 3],
        )

        # First row
        r00 = 1 - 2 * (q2 * q2 + q3 * q3)
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 1 - 2 * (q1 * q1 + q3 * q3)
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 1 - 2 * (q1 * q1 + q2 * q2)

        return th.stack(
            [
                th.stack([r00, r01, r02], dim=1),
                th.stack([r10, r11, r12], dim=1),
                th.stack([r20, r21, r22], dim=1),
            ],
            dim=1,
        )

    # Generate quaternions and convert to rotation matrices
    quaternions = points_to_quaternions(points)
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)

    # Ensure proper orthogonality (due to numerical precision)
    U, _, V = th.svd(rotation_matrices)
    rotation_matrices = th.bmm(U, V.transpose(1, 2))

    return rotation_matrices


class MDEnergyTracker:
    """Tracks energies and other observables during MD simulation"""

    def __init__(self, atoms, timestep, target_temperature):
        self.atoms = atoms  # Store reference to atoms object
        self.timestep = timestep
        self.target_temperature = target_temperature
        self.initial_temp = atoms.get_temperature()
        self.initial_kinetic = atoms.get_kinetic_energy()

        # Initialize lists to store trajectory data
        self.times = []
        self.kinetic_energies = []
        self.temperatures = []
        self.max_velocities = []

        logger.info(f"Initial temperature: {self.initial_temp:.1f} K")
        logger.info(f"Initial kinetic energy: {self.initial_kinetic:.3f} eV")

    def __call__(self):
        """Called by ASE dynamics at each observation interval"""
        kinetic = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()

        self.times.append(len(self.times) * self.timestep * 10)  # Convert to fs
        self.kinetic_energies.append(kinetic)
        self.temperatures.append(temp)
        self.max_velocities.append(
            np.max(np.linalg.norm(self.atoms.get_velocities(), axis=1))
        )

    def plot(self, save_path):
        """Plot evolution of available observables"""
        plt.figure(figsize=(12, 9))

        # Temperature subplot
        plt.subplot(3, 1, 1)
        plt.plot(self.times, self.temperatures, label="Current")
        plt.axhline(
            y=self.target_temperature,
            color="r",
            linestyle="--",
            label=f"Target ({self.target_temperature}K)",
        )
        plt.fill_between(
            self.times,
            [self.target_temperature * 0.95] * len(self.times),
            [self.target_temperature * 1.05] * len(self.times),
            color="r",
            alpha=0.1,
        )
        plt.xlabel("Time (fs)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Evolution")
        plt.legend()

        # Kinetic energy subplot
        plt.subplot(3, 1, 2)
        plt.plot(self.times, self.kinetic_energies, label="Kinetic")
        plt.xlabel("Time (fs)")
        plt.ylabel("Energy (eV)")
        plt.title("Kinetic Energy")
        plt.legend()

        # Velocity subplot
        plt.subplot(3, 1, 3)
        plt.plot(self.times, self.max_velocities)
        plt.xlabel("Time (fs)")
        plt.ylabel("Max Velocity (Å/fs)")
        plt.title("Maximum Atomic Velocity")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_stats(self):
        """Return summary statistics of the simulation"""
        return {
            "avg_temperature": np.mean(self.temperatures),
            "temp_std": np.std(self.temperatures),
            "avg_kinetic": np.mean(self.kinetic_energies),
            "kinetic_std": np.std(self.kinetic_energies),
        }

    def save_data(self, results_dir: Path):
        """Save trajectory data to files"""
        results_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data as list of dictionaries
        data = []
        for i in range(len(self.times)):
            data_point = {
                "time": self.times[i],
                "temperature": self.temperatures[i],
                "kinetic_energy": self.kinetic_energies[i],
                "max_velocity": self.max_velocities[i],
            }
            data.append(data_point)

        # Save as CSV
        csv_path = results_dir / "md_trajectory_data.csv"
        with open(csv_path, "w", newline="") as f:
            # Define the fieldnames (column headers)
            fieldnames = ["time", "temperature", "kinetic_energy", "max_velocity"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the data
            writer.writerows(data)


if __name__ == "__main__":
    run(main)
