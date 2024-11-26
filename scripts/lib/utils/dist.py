import os
from contextlib import nullcontext

import torch.distributed as dist
import torch as th

from datetime import timedelta

from loguru import logger
import numpy as np


def cleanup_dist():
    dist.destroy_process_group()


def setup_dist(rank, world_size, port="33281"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist_type = "nccl" if th.cuda.is_available() else "gloo"
    if th.cuda.is_available():
        th.cuda.set_device(rank)
    dist.init_process_group(
        dist_type,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
        init_method="env://",
    )


def setup_device(rank=0):
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True

    if th.cuda.is_available():
        th.cuda.set_device(rank)
        device = th.device("cuda", rank)
    else:
        device = th.device("cpu")

    logger.info(f"Rank {rank} running 🚀, accelerator: {device}")

    dtype = (
        "bfloat16"
        if th.cuda.is_available() and th.cuda.is_bf16_supported()
        else "float16"
    )
    dtype = "float32"  # TODO: make configurable
    logger.info(f"Data type: {dtype}")

    ptdtype = {
        "float32": th.float32,
        "bfloat16": th.bfloat16,
        "float16": th.float16,
    }[dtype]
    if device.type == "cuda":
        ctx = th.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        # nullcontext
        ctx = nullcontext()
    return ctx, device
