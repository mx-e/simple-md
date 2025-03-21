#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif uv run python
import csv
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch as th
import torch.multiprocessing as mp
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, trajectory, write
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi  # Added Bussi thermostat
from ase.md.langevin import Langevin  # Added Langevin thermostat
from ase.md.nose_hoover_chain import NoseHooverChainNVT  # Added Nose-Hoover thermostat
from ase.md.npt import NPT  # Added NPT with Nosé-Hoover
from ase.md.nvtberendsen import NVTBerendsen  # Added Berendsen thermostat
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import builds, instantiate, load_from_yaml, store
from lib.data.loaders import batch_tall, collate_fn
from lib.data.transforms import center_positions_on_center_of_mass, center_positions_on_centroid, get_random_rotations
from lib.types import Property as Props
from lib.utils.checkpoint import load_checkpoint
from lib.utils.dist import get_amp, setup_device
from lib.utils.filters import remove_net_force, remove_net_torque
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import MISSING

import wandb
from scripts.lib.utils.augmentation import (
    analyze_rotation_distribution,
    generate_equidistant_rotations,
    visualize_rotations,
)

pbuilds = partial(builds, zen_partial=True)


BOHR_TO_ANG = 0.529177249  # Bohr to Angstrom
HARTREE_TO_EV = 27.211386245988  # Hartree to eV
ANG_TO_BOHR = 1.0 / 0.529177249
FORCE_CONVERSION = HARTREE_TO_EV / BOHR_TO_ANG


def get_thermostat(
    thermostat: Literal["nose_hoover", "langevin", "bussi"],
    atoms,
    temperature: float,
    timestep: float,
    tau: float,
    nh_chain_len: int = 3,
) -> NoseHooverChainNVT | Langevin | Bussi:
    match thermostat:
        case "nose_hoover":
            return NoseHooverChainNVT(
                atoms, tchain=nh_chain_len, tloop=1, temperature_K=temperature, timestep=timestep * units.fs, tdamp=tau
            )
        case "langevin":
            return Langevin(atoms, temperature_K=temperature, timestep=timestep * units.fs, friction=1.0 / tau)
        case "bussi":
            return Bussi(atoms, temperature_K=temperature, timestep=timestep * units.fs, taut=tau)
        case _:
            raise ValueError(f"Unknown thermostat: {thermostat}")


@configure_main(extra_defaults=[])
def main(
    cfg: BaseConfig,
    timestep: float = 0.5,
    n_data_aug: int = 1,
    step_wise_random: bool = False,
    n_steps: int = 200000000,
    thermostat: Literal["nose_hoover", "langevin", "bussi"] = "langevin",
    temperature: float = 300,
    tau: float = 100.0,
    init_struct_dir: Path = "data_md",
    init_struct: str = "md17_ethanol",
    last_n_steps: int
    | None = None,  # export the last n steps of the trajectory separately and use those for the dipole spectrum
    model_run_dir: Path = MISSING,
    checkpoint_name: str = "best_model",
    dipole_model_run_dir: Path | None = None,
    dipole_checkpoint_name: str = "best_model",
    traj_log_interval: int = 10,
    ptdtype: Literal["float32", "bfloat16", "float16"] = "float32",
    remove_net_force: bool = True,
    remove_net_torque: bool = True,
    fa_outlier_log_threshold: float = 1e-3,
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    job_dir = get_hydra_output_dir()
    device, ctx = setup_device(), get_amp(ptdtype)
    model_run_conf_path = model_run_dir / ".hydra" / "config.yaml"
    model_run_conf = load_from_yaml(model_run_conf_path)
    if "ft" in model_run_conf:
        logger.info("detected fine-tuned model, loading pretrain model configuration")
        model_pt_conf_path = model_run_dir / ".hydra" / "model_pretrain_conf.yaml"
        model_conf = load_from_yaml(model_pt_conf_path)
    elif "train" in model_run_conf:
        model_conf = model_run_conf["train"]["model"]
    model = instantiate(model_conf)
    checkpoint_path = Path(model_run_dir) / "ckpts" / (checkpoint_name + ".pth")
    load_checkpoint(model, checkpoint_path)
    model.eval().to(device)

    dipole_model = None
    if dipole_model_run_dir is not None:
        dipole_model_run_conf_path = Path(dipole_model_run_dir) / ".hydra" / "config.yaml"
        dipole_model_run_conf = load_from_yaml(dipole_model_run_conf_path)
        dipole_model_conf = dipole_model_run_conf["train"]["model"]
        dipole_model = instantiate(dipole_model_conf)
        dipole_checkpoint_path = Path(dipole_model_run_dir) / "ckpts" / (dipole_checkpoint_name + ".pth")
        load_checkpoint(dipole_model, dipole_checkpoint_path)
        dipole_model.eval().to(device)

    # prepare results directory, load initial structure
    results_dir = Path(job_dir) / "md_results"
    results_dir.mkdir(exist_ok=True)
    valid_init_struct_names = [f.stem for f in Path(init_struct_dir).glob("*.xyz")]
    if init_struct not in valid_init_struct_names:
        raise ValueError(f"Invalid initial structure name: {init_struct}. Available: {valid_init_struct_names}")
    init_struct_path = Path(init_struct_dir) / (init_struct + ".xyz")
    atoms = read(init_struct_path)
    fa_outliers_log_path = results_dir / "frame_averaging_outliers.log"
    fa_outliers_log_path.parent.mkdir(exist_ok=True)
    fa_outliers_log_path.touch()

    if n_data_aug > 1:
        logger.info(f"Generating {n_data_aug} equidistant rotations...")
        rotations = generate_equidistant_rotations(n_data_aug)
        # Create visualization and save to file
        rotations_path = Path(job_dir) / "data_aug" / "rotation_visualization.png"
        rotations_path.parent.mkdir(exist_ok=True)
        visualize_rotations(rotations, rotations_path)
        stats = analyze_rotation_distribution(rotations)
        logger.info(f"Rotation Distribution Statistics: {stats}")

    eval_artifact = None
    if cfg.wandb is not None:
        eval_artifact = wandb.Artifact("md_eval", type="evaluation")
    # Run MD simulation
    traj_path = Path(job_dir) / "md_results" / "md_trajectory.traj"
    if last_n_steps is not None:
        assert last_n_steps <= n_steps, "last_n_steps must be less than or equal to n_steps"
    energy_tracker = run_md_simulation(
        atoms=atoms,
        model=model,
        dipole_model=dipole_model,
        ctx=ctx,
        device=device,
        temperature=temperature,
        timestep=timestep,
        tau=tau,
        rotations=rotations.to(device) if n_data_aug > 1 else None,
        step_wise_random_aug=step_wise_random,
        steps=n_steps,
        trajectory_file=traj_path,
        thermostat=thermostat,
        save_last_n_steps=last_n_steps,
        traj_log_interval=traj_log_interval,
        wandb_artifact=eval_artifact,
        remove_net_force=remove_net_force,
        remove_net_torque=remove_net_torque,
        fa_outlier_log_threshold=fa_outlier_log_threshold,
        fa_outliers_log_path=fa_outliers_log_path,
    )
    # Save and plot results
    energy_tracker.save_data(results_dir)
    energy_tracker.plot(results_dir / "md_analysis.png")
    if eval_artifact is not None:
        cfg.wandb.run.log_artifact(eval_artifact)

    # Print final statistics
    final_stats = energy_tracker.get_stats()
    logger.info("Simulation Statistics:")
    logger.info(f"  Average Temperature: {final_stats['avg_temperature']:.1f} ± {final_stats['temp_std']:.1f} K")
    logger.info(f"  Average Kinetic Energy: {final_stats['avg_kinetic']:.3f} ± {final_stats['kinetic_std']:.3f} eV")


def run_md_simulation(
    atoms,
    model,
    ctx,
    device,
    temperature,  # K
    tau,
    timestep,  # fs
    rotations,
    step_wise_random_aug,
    steps,
    thermostat,
    remove_net_force,
    remove_net_torque,
    trajectory_file="md_trajectory.traj",
    save_last_n_steps=None,
    dipole_model=None,
    traj_log_interval=10,
    wandb_artifact=None,
    fa_outlier_log_threshold=1e-3,
    fa_outliers_log_path=None,
) -> "MDEnergyTracker":
    # Set up calculator
    calculator = MLCalculator(
        model,
        device,
        ctx,
        rotations=rotations,
        step_wise_random_aug=step_wise_random_aug,
        remove_net_force=remove_net_force,
        remove_net_torque=remove_net_torque,
        fa_outlier_log_threshold=fa_outlier_log_threshold,
        fa_outliers_log_path=fa_outliers_log_path,
    )
    atoms.calc = calculator

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature * 2)

    dyn = get_thermostat(thermostat, atoms, temperature, timestep, tau)

    # Set up trajectory file and energy tracker
    traj = trajectory.Trajectory(trajectory_file, "w", atoms)
    energy_tracker = MDEnergyTracker(
        atoms,
        timestep,
        temperature,
        device,
        dipole_model=dipole_model,
        last_n_steps=save_last_n_steps,
        traj_log_interval=traj_log_interval,
        wandb_artifact=wandb_artifact,
    )
    xyz_trajectory = []

    def save_frame() -> None:
        frame = atoms.copy()
        frame.info["comment"] = (
            f"time={len(xyz_trajectory) * timestep:.1f}fs "
            f"temp={atoms.get_temperature():.1f}K "
            f"thermostat={thermostat.__class__.__name__}"
        )
        xyz_trajectory.append(frame)

    # Attach observers
    dyn.attach(traj.write, interval=traj_log_interval)
    dyn.attach(save_frame, interval=traj_log_interval)
    dyn.attach(energy_tracker, interval=1)

    # Run dynamics with improved monitoring
    logger.info(f"Starting MD simulation with {thermostat} thermostat for {steps} steps...")
    logger.info(f"Target temperature: {temperature}K")
    logger.info(f"Thermostat tau: {tau}")

    for i in range(steps):
        dyn.run(1)
        if i % 100 == 0:
            temp = atoms.get_temperature()
            temp_diff = abs(temp - temperature)
            temp_deviation_percent = (temp_diff / temperature) * 100

            logger.info(f"Step {i}:")
            logger.info(
                f"  Temperature: {temp:.1f}K (Target: {temperature}K, Deviation: {temp_deviation_percent:.1f}%)"
            )
            logger.info(f"  Kinetic Energy: {atoms.get_kinetic_energy():.3f} eV")
            # Check simulation stability
            if np.any(np.isnan(atoms.get_positions())) or np.any(np.isnan(atoms.get_velocities())):
                logger.error("Simulation unstable: NaN detected in positions or velocities!")
                break

            # Check molecular integrity
            distances = atoms.get_all_distances()
            max_dist = np.max(distances)
            if max_dist > 10.0:  # Å
                logger.warning(f"Simulation unstable: Atoms too far apart ({max_dist:.2f} Å)!")

    logger.info("MD simulation completed!")

    xyz_file = str(Path(trajectory_file).with_suffix(".xyz"))
    # remove cell information
    for frame in xyz_trajectory:
        frame.set_cell([0, 0, 0])
        frame.set_pbc(False)
    write(xyz_file, xyz_trajectory)

    if wandb_artifact is not None:
        wandb_artifact.add_file(xyz_file)

    if save_last_n_steps is not None and traj_log_interval == 1:
        last_n_traj = xyz_trajectory[-save_last_n_steps:]
        last_n_traj_file = str(Path(trajectory_file).with_name(f"last_{save_last_n_steps}_steps.xyz"))
        write(last_n_traj_file, last_n_traj)

    logger.info(f"Trajectory saved to {trajectory_file} and {xyz_file}")

    return energy_tracker


class MLCalculator(Calculator):
    """Calculator interfacing the trained ML model with ASE"""

    implemented_properties = ["forces"]
    not_implemented_properties = ["energy"]

    def __init__(
        self,
        model,
        device,
        ctx,
        remove_net_force,
        remove_net_torque,
        rotations=None,
        step_wise_random_aug=False,
        fa_outlier_log_threshold=1e-3,
        fa_outliers_log_path=None,
    ) -> None:
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
        self.remove_net_force = remove_net_force
        self.remove_net_torque = remove_net_torque
        self.fa_outlier_log_threshold = fa_outlier_log_threshold
        self.fa_outliers_log_path = fa_outliers_log_path

    def convert_atoms_to_model_input(self, atoms, rotations) -> tuple[dict, th.Tensor]:
        """Convert ASE Atoms object to model input format"""
        positions_bohr = atoms.positions * ANG_TO_BOHR
        data = {
            "positions": positions_bohr,
            "atomic_numbers": atoms.numbers,
            "charge": 0,
            "multiplicity": 1,
        }

        data = self.collate_fn([data])

        reverse_rotations = None
        if rotations is not None:
            for k, v in data.items():
                expand_shape = (len(rotations), *tuple(-1 for _ in range(v.dim() - 1)))
                data[k] = v.expand(expand_shape)
            if self.step_wise_random_aug:
                random_offset_rotation = get_random_rotations(1, self.device).expand(len(rotations), -1, -1)
                rotations_step = th.bmm(rotations, random_offset_rotation)
            else:
                rotations_step = rotations

            data[Props.positions] = th.bmm(data[Props.positions], rotations_step).squeeze()
            reverse_rotations = rotations_step.transpose(1, 2)
        return data, reverse_rotations

    @th.no_grad()
    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        if properties is None:
            properties = ["forces"]
        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert atoms to model input
        data, reverse_rotations = self.convert_atoms_to_model_input(atoms, self.rotations)

        # Get predictions from model
        with self.ctx:
            outputs = self.model(data)

        n_nodes = (data[Props.atomic_numbers] != 0).sum(dim=1)  # (b,)
        if self.remove_net_force:
            outputs[Props.forces] = remove_net_force(forces=outputs[Props.forces], n_nodes=n_nodes)

        if self.remove_net_torque:
            outputs[Props.forces] = remove_net_torque(
                positions=data[Props.positions], forces=outputs[Props.forces], n_nodes=n_nodes
            )

        if self.rotations is not None:
            outputs[Props.forces] = th.bmm(outputs[Props.forces], reverse_rotations)  # (n_rotations, n_atoms, 3)
            check_rotations_for_outliers_and_log(
                forces=outputs[Props.forces],
                fa_outliers_log_path=self.fa_outliers_log_path,
                outlier_log_threshold=self.fa_outlier_log_threshold,
            )
            mean_force_predictions = outputs[Props.forces].mean(dim=0)
        else:
            mean_force_predictions = outputs[Props.forces].squeeze()
        mean_force_predictions *= FORCE_CONVERSION

        # Convert outputs to calculator format
        self.results = {
            "forces": mean_force_predictions.squeeze().cpu().numpy(),
        }


def check_rotations_for_outliers_and_log(
    forces: th.Tensor,  # (n_rotations, n_atoms, 3)
    fa_outliers_log_path: Path,
    outlier_log_threshold: float,
) -> None:
    if fa_outliers_log_path is None:
        return
    mean_forces = forces.mean(dim=0, keepdim=True)  # (1, n_atoms, 3)
    dists = (forces - mean_forces).norm(dim=-1)  # (n_rotations, n_atoms)
    outliers = dists > outlier_log_threshold
    if outliers.any():
        logger.warning(f"Frame averaging outliers detected: {outliers.sum()} / {len(outliers.flatten())}")
        max_outlier = th.max(dists)
        logger.warning(f"Max outlier distance from mean prediction: {max_outlier:.2e}")
        outliers = th.nonzero(outliers, as_tuple=True)
        with fa_outliers_log_path.open("a") as f:
            f.write("--- Outliers Detected ---\n")
            f.write(f"DISTS: {dists}\n")
            f.write(f"OUTLIERS: {outliers}\n")
            f.write(f"MAX OUTLIER DIST: {max_outlier}\n")
            f.write("\n")


class MDEnergyTracker:
    """Tracks energies and other observables during MD simulation"""

    def __init__(
        self,
        atoms,
        timestep,
        target_temperature,
        device,
        dipole_model=None,
        last_n_steps=0,
        traj_log_interval=1,
        wandb_artifact=None,
    ) -> None:
        self.atoms = atoms  # Store reference to atoms object
        self.timestep = timestep
        self.target_temperature = target_temperature
        self.initial_temp = atoms.get_temperature()
        self.initial_kinetic = atoms.get_kinetic_energy()
        self.dipole_model = dipole_model
        self.last_n_steps = last_n_steps
        self.traj_log_interval = traj_log_interval
        self.wandb_artifact = wandb_artifact

        # Initialize lists to store trajectory data
        self.times = []
        self.kinetic_energies = []
        self.temperatures = []
        self.max_velocities = []
        self.potential_energies = []
        self.total_energies = []
        self.dipole_moments = []

        # For potential energy integration
        self.integrated_energy = 0.0
        self.last_positions = None
        self.last_forces = None

        pre_batch_preprocessors = [center_positions_on_center_of_mass]
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

        logger.info(f"Initial temperature: {self.initial_temp:.1f} K")
        logger.info(f"Initial kinetic energy: {self.initial_kinetic:.3f} eV")

    def reset_potential_energy(self) -> None:
        """Reset the potential energy integration"""
        self.integrated_energy = 0.0
        self.last_positions = None
        self.last_forces = None

    def evaluate_dipole_model(self) -> np.ndarray | None:
        """Evaluate dipole moment using the dipole model"""
        if self.dipole_model is None:
            return None

        # Convert positions to Bohr
        positions_bohr = self.atoms.positions * ANG_TO_BOHR

        # Prepare data dictionary
        data = {
            "positions": positions_bohr,
            "atomic_numbers": self.atoms.numbers,
            "charge": self.atoms.get_initial_charges().sum(),
            "multiplicity": 2 * abs(self.atoms.get_initial_magnetic_moments().sum()) + 1,
        }

        # Get predictions from model
        with th.no_grad():
            data = self.collate_fn([data])
            outputs = self.dipole_model(data)
            # Model returns in e*Bohr, convert to Debye (1 e*Bohr ≈ 2.542 Debye)
            dipole = outputs[Props.dipole].squeeze().cpu().numpy() * 2.541746

        return dipole

    def __call__(self) -> None:
        """Called by ASE dynamics at each observation interval"""
        # Get basic observables
        kinetic = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        forces = self.atoms.get_forces()
        current_positions = self.atoms.get_positions()

        # Calculate potential energy through force integration
        if self.last_positions is not None:
            displacement = current_positions - self.last_positions
            avg_forces = 0.5 * (forces + self.last_forces) if hasattr(self, "last_forces") else forces
            # Scale energy change by timestep (fs)
            delta_energy = -np.sum(avg_forces * displacement) / self.timestep  # F·dx/dt
            self.integrated_energy += delta_energy

        self.last_positions = current_positions.copy()
        self.last_forces = forces.copy()

        # Get dipole if model is available
        if self.dipole_model is not None:
            dipole = self.evaluate_dipole_model()
            self.dipole_moments.append(dipole)

        # Store all data
        self.times.append(len(self.times) * self.timestep)  # Convert to fs
        self.kinetic_energies.append(kinetic)
        self.temperatures.append(temp)
        self.max_velocities.append(np.max(np.linalg.norm(self.atoms.get_velocities(), axis=1)))
        self.potential_energies.append(self.integrated_energy)
        self.total_energies.append(kinetic + self.integrated_energy)

    def compute_ir_spectrum(self, max_freq=4000) -> tuple[np.ndarray, np.ndarray]:
        """Compute IR spectrum from dipole moment time series.
        Uses direct Fourier transform of dipole moments.
        """
        if not self.dipole_moments:
            return None, None

        # Get dipole time series
        if self.last_n_steps > 0 and self.last_n_steps < len(self.dipole_moments):
            dipoles = np.array(self.dipole_moments[-self.last_n_steps :])
            logger.info(f"Using last {self.last_n_steps} steps for IR spectrum")
        else:
            dipoles = np.array(self.dipole_moments)
            logger.info(f"Using all {len(dipoles)} steps for IR spectrum")

        # Time step and number of points
        dt = self.timestep * 1e-15  # Convert fs to seconds
        # Convert fs to picoseconds
        n_points = len(dipoles)

        # Get frequency axis in Hz
        freqs_hz = np.fft.fftfreq(n_points, dt)

        # Convert to wavenumbers (cm^-1)
        c = 29979245800  # Speed of light in cm/s
        freqs = freqs_hz / c  # Convert to cm^-1
        # Compute FFT and power spectrum
        dipole_norm = np.linalg.norm(dipoles, axis=1)
        dipole_norm -= np.mean(dipole_norm)
        dipole_norm = dipole_norm * np.hanning(len(dipole_norm))
        spectrum = np.fft.fft(dipole_norm)

        # Truncate to max frequency
        mask = freqs <= max_freq
        mask *= freqs >= 100
        freqs = freqs[mask]
        spectrum = spectrum[mask]

        spectrum = np.abs(spectrum)
        spectrum = spectrum / np.max(spectrum)

        logger.info(f"Computed spectrum: {len(spectrum)} points from {freqs[0]:.1f} to {freqs[-1]:.1f} cm^-1")

        return freqs, spectrum

    def plot(self, save_path) -> None:
        n_plots = 6 if self.dipole_model is not None else 4  # One extra subplot for dipole norm
        plt.figure(figsize=(12, 3 * n_plots))

        # Temperature subplot
        plt.subplot(n_plots, 1, 1)
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

        # Energy components subplot
        plt.subplot(n_plots, 1, 2)
        plt.plot(self.times, self.kinetic_energies, label="Kinetic")
        plt.plot(self.times, self.potential_energies, label="Potential")
        plt.plot(self.times, self.total_energies, label="Total")
        plt.xlabel("Time (fs)")
        plt.ylabel("Energy (eV)")
        plt.title("Energy Components")
        plt.legend()

        # Energy conservation subplot
        plt.subplot(n_plots, 1, 3)
        energy_drift = np.array(self.total_energies) - self.total_energies[0]
        plt.plot(self.times, energy_drift)
        plt.xlabel("Time (fs)")
        plt.ylabel("ΔE (eV)")
        plt.title("Total Energy Drift")

        # Velocity subplot
        plt.subplot(n_plots, 1, 4)
        plt.plot(self.times, self.max_velocities)
        plt.xlabel("Time (fs)")
        plt.ylabel("Max Velocity (Å/fs)")
        plt.title("Maximum Atomic Velocity")

        # Dipole subplots (if available)
        if self.dipole_model is not None and self.dipole_moments:
            # Dipole components
            plt.subplot(n_plots, 1, 5)
            dipole_array = np.array(self.dipole_moments)
            plt.plot(self.times, dipole_array[:, 0], label="x")
            plt.plot(self.times, dipole_array[:, 1], label="y")
            plt.plot(self.times, dipole_array[:, 2], label="z")
            plt.xlabel("Time (fs)")
            plt.ylabel("Dipole Components (Debye)")
            plt.title("Molecular Dipole Components")
            plt.legend()

            # Dipole magnitude
            plt.subplot(n_plots, 1, 6)
            plt.plot(self.times, np.linalg.norm(dipole_array, axis=1), color="black")
            plt.xlabel("Time (fs)")
            plt.ylabel("Dipole Magnitude (Debye)")
            plt.title("Molecular Dipole Magnitude")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        if self.wandb_artifact is not None:
            self.wandb_artifact.add_file(save_path)

    def get_stats(self) -> dict:
        """Return summary statistics of the simulation"""
        stats = {
            "avg_temperature": np.mean(self.temperatures),
            "temp_std": np.std(self.temperatures),
            "avg_kinetic": np.mean(self.kinetic_energies),
            "kinetic_std": np.std(self.kinetic_energies),
            "avg_potential": np.mean(self.potential_energies),
            "potential_std": np.std(self.potential_energies),
            "total_energy_drift": self.total_energies[-1] - self.total_energies[0],
        }

        if self.dipole_model is not None and self.dipole_moments:
            dipole_array = np.array(self.dipole_moments)
            dipole_magnitudes = np.linalg.norm(dipole_array, axis=1)
            stats.update(
                {
                    "avg_dipole_magnitude": np.mean(dipole_magnitudes),
                    "dipole_magnitude_std": np.std(dipole_magnitudes),
                    "max_dipole_magnitude": np.max(dipole_magnitudes),
                    "min_dipole_magnitude": np.min(dipole_magnitudes),
                }
            )

        return stats

    def save_data(self, results_dir: Path) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save main trajectory data
        data = []
        for i in range(len(self.times)):
            data_point = {
                "time": self.times[i],
                "temperature": self.temperatures[i],
                "kinetic_energy": self.kinetic_energies[i],
                "potential_energy": self.potential_energies[i],
                "total_energy": self.total_energies[i],
                "max_velocity": self.max_velocities[i],
            }
            data.append(data_point)

        # Save main trajectory CSV
        logger.info("Saving main trajectory data...")
        csv_path = results_dir / "md_trajectory_data.csv"
        with csv_path.open("w", newline="") as f:
            fieldnames = ["time", "temperature", "kinetic_energy", "potential_energy", "total_energy", "max_velocity"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        if self.wandb_artifact is not None:
            self.wandb_artifact.add_file(csv_path)
        logger.info(f"Saved trajectory data to {csv_path}")

        # Save dipole moments as separate CSV if available
        if self.dipole_model is not None and self.dipole_moments:
            logger.info("Saving dipole trajectory data...")
            dipole_path = results_dir / "dipole_trajectory.csv"
            dipole_data = []
            for i, dipole in enumerate(self.dipole_moments):
                dipole_data.append(
                    {
                        "time": self.times[i],
                        "dipole_x": dipole[0],
                        "dipole_y": dipole[1],
                        "dipole_z": dipole[2],
                        "dipole_magnitude": np.linalg.norm(dipole),
                    }
                )

            with dipole_path.open("w", newline="") as f:
                fieldnames = ["time", "dipole_x", "dipole_y", "dipole_z", "dipole_magnitude"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dipole_data)
            if self.wandb_artifact is not None:
                self.wandb_artifact.add_file(dipole_path)
            logger.info(f"Saved dipole trajectory to {dipole_path}")

            # Compute and save IR spectrum
            logger.info("Computing IR spectrum...")
            freqs, spectrum = self.compute_ir_spectrum()
            if freqs is not None and len(freqs) > 1:
                # Save spectrum data to CSV with high precision
                spectrum_path = results_dir / "ir_spectrum.csv"
                with spectrum_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frequency_cm-1", "intensity"])
                    # Write all points with high precision
                    for freq, inten in zip(freqs, spectrum, strict=True):
                        writer.writerow([f"{freq:.6f}", f"{inten:.6f}"])
                logger.info(f"Saved IR spectrum data to {spectrum_path}")
                # Plot IR spectrum
                logger.info("Plotting IR spectrum...")
                plt.figure(figsize=(12, 6))
                # Plot with inverted axes
                plt.plot(freqs, spectrum)

                # Customize axes
                plt.xlabel("Wavenumber (cm⁻¹)")
                plt.ylabel("Intensity (arb. units)")
                plt.title("Dipole Spectrum")

                # Set axis ranges
                plt.xlim(4000, 100)  # Wavenumbers decrease left to right
                plt.ylim(1.0, -0.02)

                # Make plot cleaner
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)

                plt.tight_layout()
                plot_path = results_dir / "ir_spectrum.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                if self.wandb_artifact is not None:
                    self.wandb_artifact.add_file(plot_path)
                logger.info(f"Saved IR spectrum plot to {plot_path}")
            else:
                logger.warning("Could not compute IR spectrum - insufficient data points")


if __name__ == "__main__":
    run(main)
