#! /usr/bin/env -S apptainer exec --nv --bind /temp:/temp_data --bind /home/bbdc2/quantum/max/:/data container.sif python
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from conf.base_conf import BaseConfig, configure_main
from hydra_zen import builds, instantiate, load_from_yaml, store
from hydra_zen.typing import Partial
from lib.data.loaders import get_loaders
from lib.datasets import get_qcml_dataset, get_rmd17_dataset
from lib.ema import EMAModel
from lib.loss import LossModule
from lib.lr_scheduler import LRScheduler, get_lr_scheduler
from lib.models import PairEncoder, get_pair_encoder_pipeline_config
from lib.models.pair_encoder import NodeLevelRegressionHead
from lib.train_loop import Predictor, evaluate, train_loop
from lib.types import PipelineConfig, Split
from lib.utils.checkpoint import load_checkpoint, save_checkpoint
from lib.utils.dist import cleanup_dist, get_amp, setup_device, setup_dist
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.run import run
from loguru import logger
from omegaconf import MISSING
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

pbuilds = partial(builds, zen_partial=True)
pbuilds_full = partial(builds, zen_partial=True, populate_full_signature=True)

p_optim = pbuilds(
    th.optim.AdamW,
    weight_decay=1e-7,
)
p_ema = pbuilds(
    EMAModel,
    decay=0.9997,
)
p_no_scheduler = pbuilds(get_lr_scheduler)
p_cosine_scheduler = pbuilds(
    get_lr_scheduler,
    scheduler_type="none",
    warmup_steps=0,
    min_lr=1e-7,
)
loss_module_dipole = builds(
    LossModule,
    targets=["dipole"],
    loss_types={"dipole": "euclidean"},
    metrics={"dipole": ["mae", "mse", "euclidean"]},
    compute_metrics_train=False,
)
loss_module_forces = builds(
    LossModule,
    targets=["forces"],
    loss_types={"forces": "euclidean"},
    metrics={"forces": ["mae", "mse", "euclidean"]},
    compute_metrics_train=False,
)
pair_encoder_data_config = builds(
    get_pair_encoder_pipeline_config,
    augmentation_mult=2,
    random_rotation=True,
    random_reflection=True,
    center_positions=True,
    dynamic_batch_size_cutoff=29,
    include_dipole=False,
)
ft_loop = pbuilds(
    train_loop,
    log_interval=5,
    eval_interval=20,
    save_interval=50000,
    eval_samples=50000,
    clip_grad=1.0,
    ptdtype="float32",
)

qcml_data = pbuilds(
    get_qcml_dataset,
    data_dir="./data_ar",
    dataset_name="qcml_unified_fixed_split_by_smiles",
    dataset_version="1.0.0",
    copy_to_temp=True,
)

md17_aspirin = pbuilds(
    get_rmd17_dataset,
    data_dir="./data",
    molecule_name="aspirin",
    splits={"train": 0.8, "val": 0.1, "test": 0.1},
)
dataset_store = store(group="ft/dataset")
dataset_store(qcml_data, name="qcml")
dataset_store(md17_aspirin, name="md17_aspirin")


def finetune(
    rank: int,
    port: str,
    world_size: int,
    cfg: BaseConfig,
    pretrain_model_dir: Path,
    checkpoint_name: str = "best_model",
    dataset=MISSING,
    optimizer: Partial[th.optim.Optimizer] = p_optim,
    train_loop: Partial[callable] = ft_loop,
    finetune_type: Literal["head_only", "full"] = "full",
    train_size: Literal["zero_shot", "few_shot", "full"] = "few_shot",
    few_shot_size: int = 1000,
    batch_size: int = 128,
    total_steps: int = 1000,
    final_eval_samples: int = 50000,
    lr: float = 5e-5,
    grad_accum_steps: int = 1,
    lr_scheduler: Partial[callable] | None = None,  # None = No schedule
    loss: LossModule | None = loss_module_forces,  # None = Same as pretrain
    pipeline_conf: PipelineConfig | None = pair_encoder_data_config,  # None = Same as pretrain
    ema: Partial[EMAModel] | None = None,  # None = No EMA
) -> None:
    setup_dist(rank, world_size, port=port)
    try:
        device = setup_device(rank)

        # get model + loss
        config_path = pretrain_model_dir / ".hydra" / "config.yaml"
        conf = load_from_yaml(config_path)
        model_conf = conf["train"]["model"]
        model = instantiate(model_conf)
        if loss is None:
            loss = instantiate(conf["train"]["loss"])
        model = Predictor(model, loss).to(device)
        ddp_args = {
            "device_ids": ([rank] if cfg.runtime.device == "cuda" else None),
        }
        model = DDP(model, **ddp_args)
        checkpoint_path = pretrain_model_dir / "ckpts" / f"{checkpoint_name}.pth"
        load_checkpoint(model.module.encoder, checkpoint_path)
        dist.barrier()
        if ema is not None and rank == 0:
            ema = ema(model.module, device=device)
            logger.info(f"Using EMA with decay {ema.decay}")
        else:
            ema = None
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params}")

        # handle cross-objective finetuning
        loss_pretrain = instantiate(conf["train"]["loss"])
        if loss_pretrain.targets != loss.targets:
            logger.info(
                f"Cross-objective finetuning detected: pretrain loss targets {loss_pretrain.targets} vs finetune loss targets {loss.targets}"
            )
            if not isinstance(model.module.encoder, PairEncoder):
                raise ValueError("Cross-objective finetuning only supported for PairEncoder models")
            logger.info(f"Modifying model to output {loss.targets} instead of {loss_pretrain.targets}")
            current_heads = model.module.encoder.heads
            new_heads = {target: head for target, head in current_heads.items() if target in loss.targets}
            new_head_targets = [target for target in loss.targets if target not in loss_pretrain.targets]
            for target in new_head_targets:
                new_heads[str(target)] = NodeLevelRegressionHead(
                    target=target,
                    embd_dim=model_conf.embd_dim,
                    cls_token=model_conf.cls_token,
                    activation=model_conf.activation,
                    head_dropout=model_conf.head_dropout,
                    project_down=model_conf.head_project_down,
                ).to(device)
            model.module.encoder.heads = nn.ModuleDict(new_heads)
            for target in new_head_targets:  # dataloading pipeline needs to get new targets
                pipeline_conf.needed_props.append(target)
            model = DDP(model.module, **ddp_args)  # rewrap model after modification

        match finetune_type:  # optim + lr_scheduler
            case "head_only":
                optim = optimizer(model.module.encoder.heads.parameters())
            case "full":
                optim = optimizer(model.parameters(), lr=lr)

        lr_scheduler = (
            lr_scheduler(optim, lr, lr_decay_steps=total_steps) if lr_scheduler is not None else LRScheduler(optim, lr)
        )

        # data + loaders
        data = dataset(rank)
        if pipeline_conf is None:
            try:
                pipeline_conf = instantiate(conf["train"]["pipeline_conf"])
                logger.info("Loaded pipeline config from pretraining config")
            except KeyError as e:
                raise ValueError(
                    "No pipeline config found in pretrain config - this might be an old checkpoint, please specify manually"
                ) from e
        loaders = get_loaders(
            rank=rank,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            world_size=world_size,
            device=device,
            dataset_splits=data,
            pipeline_config=pipeline_conf,
            restrict_train_size=few_shot_size if train_size == "few_shot" else None,
            num_workers=0,
        )
        logger.info(f"Train samples: {len(loaders[Split.train].dataset)}")
        logger.info(f"Val samples: {len(loaders[Split.val].dataset)}")
        logger.info(f"Test samples: {len(loaders[Split.test].dataset)}")
        if train_size != "zero_shot":
            logger.info(f"Finetuning on {train_size} data with {few_shot_size} samples")
            final_model = train_loop(
                rank=rank,
                model=model,
                loaders=loaders,
                optimizer=optim,
                save_dir=cfg.runtime.out_dir / "ckpts",
                start_step=0,
                total_steps=total_steps,
                grad_accum_steps=grad_accum_steps,
                lr_scheduler=lr_scheduler,
                ema=ema,
                wandb=cfg.wandb,
            )
        else:
            logger.info("Zero-shot finetuning - skipping training")
            final_model = model

        # evaluation
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            amp = get_amp("float32")
            val_results = evaluate(
                model=final_model,
                loader=loaders[Split.val],
                ctx=amp,
                ema=ema,
                eval_samples=final_eval_samples,
            )
            test_results = evaluate(
                model=final_model,
                loader=loaders[Split.test],
                ctx=amp,
                ema=ema,
                eval_samples=final_eval_samples,
            )
            # save model and results
            if cfg.wandb is not None:
                results_data = [[metric_name, metric, "val"] for metric_name, metric in val_results.items()] + [
                    [metric_name, metric, "test"] for metric_name, metric in test_results.items()
                ]
                results_table = wandb.Table(
                    columns=["metric", "value", "split"],
                    data=results_data,
                )

                eval_artifact = wandb.Artifact(f"eval_results_{wandb.run.id}", type="evaluation")
                eval_artifact.add(results_table, "results_table")
                eval_artifact.add(wandb.Data(data=val_results, type="val_results"), "val_results")
                eval_artifact.add(wandb.Data(data=test_results, type="test_results"), "test_results")
                cfg.wandb.run.log_artifact(eval_artifact)

                for metric_name, metric in val_results.items():
                    cfg.wandb.run.summary[f"final_val/{metric_name}"] = metric
                for metric_name, metric in test_results.items():
                    cfg.wandb.run.summary[f"final_test/{metric_name}"] = metric

            save_checkpoint(
                final_model.module.encoder,
                optimizer,
                total_steps,
                cfg.runtime.out_dir / "ckpts" / "model_final.pth",
                ema,
            )
    finally:
        cleanup_dist()


p_ft_func = pbuilds_full(finetune)


@configure_main(extra_defaults=[{"ft/dataset": "qcml"}])
def main(
    cfg: BaseConfig,  # you must keep this argument
    pretrain_model_dir: str,
    ft: Partial[callable] = p_ft_func,
) -> None:
    logger.info(f"Running with base config: {cfg}")
    mp.set_start_method("spawn", force=True)
    world_size = cfg.runtime.n_gpu if th.cuda.is_available() else 1
    logger.info(f"Running {world_size} process(es)")
    random_port = str(np.random.randint(20000, 50000))
    cfg.runtime.out_dir = get_hydra_output_dir()

    if world_size > 1:
        th.multiprocessing.spawn(
            ft,
            args=(random_port, world_size, cfg),
            nprocs=world_size,
            join=True,
            pretrain_model_dir=Path(pretrain_model_dir),
        )
    else:
        ft(rank=0, port=random_port, world_size=1, cfg=cfg, pretrain_model_dir=Path(pretrain_model_dir))


if __name__ == "__main__":
    dataset_store.add_to_hydra_store()
    run(main)
