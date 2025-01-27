#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif python
"""
Single-GPU Benchmark Script:
 - Loads & compiles a model.
 - Evaluates metrics & measures throughput.
 - Optionally computes FLOPs using torch.profiler.
 - Logs the distribution of inference times to W&B as a histogram.
"""

from functools import partial
import json
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch as th
from loguru import logger
import torch_tensorrt

# Hydra / OmegaConf / Hydra-Zen
from hydra_zen import builds, instantiate, load_from_yaml, store
from hydra_zen.typing import Partial
from omegaconf import MISSING

import wandb

# Project-specific imports
from conf.base_conf import BaseConfig, configure_main
from scripts.lib.datasets.utils import get_split_by_molecule_name
from lib.data.loaders import get_loaders
from lib.datasets import (
    get_md17_22_dataset,
    get_rmd17_dataset,
    get_ko2020_dataset,
    get_qcml_dataset,
    get_qm7x_dataset,
    get_qm7x_pbe0_dataset,
)
from lib.ema import EMAModel
from lib.loss import LossModule
from lib.lr_scheduler import LRScheduler
from lib.models import PairEncoder
from lib.models.pair_encoder import NodeLevelRegressionHead
from lib.train_loop import Predictor, evaluate
from lib.types import PipelineConfig, Split
from lib.utils.checkpoint import load_checkpoint
from lib.utils.dist import get_amp, setup_device
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from lib.models import PairEncoder, get_pair_encoder_pipeline_config

# ------------------------------------------------------------------------------
# 1) Store dataset configs (Hydra group="bench.dataset")
# ------------------------------------------------------------------------------

pbuilds = partial(builds, zen_partial=True)  # For convenience

pair_encoder_data_config = builds(
    get_pair_encoder_pipeline_config,
    augmentation_mult=2,
    random_rotation=True,
    random_reflection=True,
    center_positions=True,
    dynamic_batch_size_cutoff=10000,
    include_dipole=False,
)

md17_aspirin = pbuilds(
    get_md17_22_dataset,
    data_dir="/temp_data",
    molecule_name="aspirin",
)

rmd17_aspirin = pbuilds(
    get_rmd17_dataset,
    data_dir="/temp_data",
    molecule_name="aspirin",
)

benchmark_ds_store = store(group="bench.dataset")
benchmark_ds_store(md17_aspirin, name="md17_aspirin")
benchmark_ds_store(rmd17_aspirin, name="rmd17_aspirin")

# ------------------------------------------------------------------------------
# 2) Define a helper for FLOPs measurement (using PyTorch Profiler)
# ------------------------------------------------------------------------------

def measure_flops(model: th.nn.Module, sample_batch: dict, amp) -> float:
    """
    Performs a single forward pass under torch.profiler to estimate FLOPs.

    Returns:
        flops (float): total floating-point operations measured during the forward pass.
    """
    import torch.profiler

    model.eval()
    with th.inference_mode(), amp:
        # jit model
        for i in range(10):
            model(sample_batch)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            model(sample_batch)

    total_flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            total_flops += evt.flops

    return total_flops

# ------------------------------------------------------------------------------
# 3) Single-GPU benchmark function
# ------------------------------------------------------------------------------

def benchmark(
    cfg: BaseConfig,
    pretrain_model_dir: Path,
    checkpoint_name: str = "best_model",
    dataset=MISSING,
    compile_backend: Literal["inductor", "tensorrt", "eager"] = "inductor",
    batch_size: int = 500,
    measure_batches: int = 500,
    warmup_steps: int = 50,
    compute_flops: bool = False,
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    pipeline_conf: PipelineConfig | None = pair_encoder_data_config,
) -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.benchmark = True

    config_path = pretrain_model_dir / ".hydra" / "config.yaml"
    logger.info(f"Loading training config from {config_path}")
    conf = load_from_yaml(config_path)
    model_conf = conf["train"]["model"]

    model = instantiate(model_conf)
    checkpoint_path = Path(pretrain_model_dir) / "ckpts" / (checkpoint_name + ".pth")
    load_checkpoint(model, checkpoint_path)
    model.eval().to(device)

    ckpt_path = pretrain_model_dir / "ckpts" / f"{checkpoint_name}.pth"
    logger.info(f"Loading checkpoint from {ckpt_path}")
    load_checkpoint(model, ckpt_path)  # Make sure `Predictor` has `encoder` attribute

    if compile_backend != "eager":
        logger.info(f"Compiling model with backend={compile_backend}")
        model = th.compile(model, backend=compile_backend)
    else:
        logger.info("Using PyTorch Eager (no torch.compile).")

    data_splits = {"train": 2, "test": batch_size*(measure_batches+warmup_steps+1)}
    dataset_splits = dataset(splits=data_splits, rank=0)
    if pipeline_conf is None:
        try:
            pipeline_conf = instantiate(conf["train"]["pipeline_conf"])
            logger.info("Loaded pipeline config from pretraining config")
        except KeyError as e:
            raise ValueError(
                "No pipeline config found in pretrain config - this might be an old checkpoint, please specify manually"
            ) from e
    loaders = get_loaders(
        rank=0,
        device=device,
        world_size=1,
        dataset_splits=dataset_splits,
        batch_size=batch_size,
        num_workers=0,
        pipeline_config=pipeline_conf,
        grad_accum_steps=1,
    )
    test_loader = loaders[Split.test]
    num_test_samples = len(test_loader.dataset)
    logger.info(f"Loaded {num_test_samples} test samples "
                f"for dataset={dataset.keywords.get('molecule_name','') or dataset.keywords.get('dataset_name')}")

    amp = get_amp(dtype)
    if compute_flops:
        logger.info("Measuring FLOPs via torch.profiler (single batch).")
        sample_batch_iter = iter(test_loader)
        sample_batch = next(sample_batch_iter)
        flops = measure_flops(model, sample_batch, amp=amp)
        logger.info(f"Estimated FLOPs for a single forward pass: {flops:,.0f}")

    times = []
    seen = 0
    if measure_batches > 0:
        logger.info(f"Measuring throughput for {measure_batches} batches (warmup={warmup_steps})...")
        model.eval()
        with th.inference_mode(), amp:
            for i, batch in enumerate(test_loader):
                if i >= warmup_steps:
                    start = th.cuda.Event(enable_timing=True)
                    end = th.cuda.Event(enable_timing=True)
                    start.record()

                _ = model(batch)

                if i >= warmup_steps:
                    end.record()
                    th.cuda.synchronize()
                    times.append(start.elapsed_time(end) / 1e3)
                    seen += 1

                if seen >= measure_batches:
                    break

        avg_time = np.mean(times) if times else 0.0
        logger.info(f"Measured {seen} batches in {avg_time:.2f} sec")
        throughput = (seen / avg_time) if avg_time > 0 else 0.0
        logger.info(f"Throughput: {throughput:.2f} batches/sec, "
                    f"Avg Latency (batch): {avg_time * 1000:.2f} ms")

    # 10) Log results & histogram to W&B (if cfg.wandb=True)
    if cfg.wandb and times:
            wandb.log({"inference_timings": wandb.Histogram(times)})
            wandb.run.summary["throughput"] = throughput
            wandb.run.summary["avg_latency"] = avg_time
            if compute_flops:
                wandb.run.summary["flops"] = flops

    # End W&B run
    wandb.finish()

# ------------------------------------------------------------------------------
# 4) Hydra entry point (single-GPU main)
# ------------------------------------------------------------------------------

p_bench = builds(
    benchmark,
    populate_full_signature=True,
    zen_partial=True,
)

@configure_main(extra_defaults=[{"bench.dataset": "md17_aspirin"}])
def main(
    cfg: BaseConfig,
    pretrain_model_dir: str,
    bench: Partial[callable] = p_bench,
) -> None:
    """
    Hydra main for single-GPU benchmarking a model on a dataset.

    Usage:
      python single_gpu_benchmark.py pretrain_model_dir=/path/to/run
      # or override dataset:    bench.dataset=qcml
      # or override FLOPs:      bench.compute_flops=true
      # or override backend:    bench.compile_backend=tensorrt
    """
    logger.info(f"Running with base config: {cfg}")
    cfg.runtime.out_dir = get_hydra_output_dir()
    print(th.compiler.list_backends())

    bench(
        cfg=cfg,
        pretrain_model_dir=Path(pretrain_model_dir),
    )

if __name__ == "__main__":
    benchmark_ds_store.add_to_hydra_store()
    run(main)
