from functools import partial

import torch as th
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.types import (
    Property as Props,
    PropertyType as PropsType,
    property_dims,
    property_type,
    property_dtype,
    edge_props_to_local_map,
    PipelineConfig,
    DatasetSplits,
    Split,
)


def get_loaders(
    rank,
    batch_size,
    grad_accum_steps,
    world_size,
    device,
    dataset_splits: DatasetSplits,
    pipeline_config: PipelineConfig,
):
    assert all(
        prop in dataset_splits.dataset_props for prop in pipeline_config.needed_props
    ), f"Props needed by the model not present in the dataset {list(filter(lambda x: x not in dataset_splits.dataset_props, pipeline_config.needed_props))}"
    needed_props = {
        prop: dataset_splits.dataset_props[prop]
        for prop in pipeline_config.needed_props
    }
    # shuffle only train data
    samplers = {
        k: DistributedSampler(
            v, shuffle=(k == Split.train), num_replicas=world_size, rank=rank
        )
        for k, v in dataset_splits.splits.items()
    }
    batch_func = batch_flat if pipeline_config.collate_type == "flat" else batch_tall
    assert (
        batch_size % (world_size * grad_accum_steps * pipeline_config.batch_size_impact)
        == 0
    ), "Batch size must be divisible by world_size * grad_accum_steps * pipeline_config.batch_size_impact"
    effective_batch_size = int(
        batch_size
        // (pipeline_config.batch_size_impact * world_size * grad_accum_steps)
    )
    loaders = {
        k: DataLoader(
            ds,
            batch_size=(effective_batch_size if k == Split.train else batch_size),
            sampler=samplers[k],
            collate_fn=partial(
                collate_fn,
                props=needed_props,
                batch_func=batch_func,
                pre_batch_preprocessors=(
                    pipeline_config.pre_collate_processors
                    if k == Split.train
                    else pipeline_config.pre_collate_processors_val
                ),
                post_batch_preprocessors=(
                    pipeline_config.post_collate_processors
                    if k == Split.train
                    else pipeline_config.post_collate_processors_val
                ),
                device=device,
            ),
            num_workers=4,
        )
        for k, ds in dataset_splits.splits.items()
    }

    return loaders


def batch_tall(batch, props: list[Props], n_atoms):
    max_atoms = th.max(n_atoms).item()
    out = {Props.mask: th.zeros(len(batch), max_atoms, dtype=bool)}

    for prop in props:
        if property_type[prop] == PropsType.mol_wise:
            out[prop] = th.stack([sample[prop] for sample in batch])
        elif property_type[prop] == PropsType.atom_wise:
            out[prop] = th.zeros(len(batch), max_atoms, property_dims[prop]).squeeze(-1)
        else:
            raise NotImplementedError(
                f"Props type {property_type[prop]} not supported for tall batching"
            )

    for prop in props:
        if property_type[prop] == PropsType.atom_wise:
            for i, sample in enumerate(batch):
                out[prop][i, : n_atoms[i]] = sample[prop]
                out[Props.mask][i, : n_atoms[i]] = 1
    return out


def batch_flat(batch, props: list[Props], n_atoms):
    out = {}
    at_cumsum = th.cat([th.zeros(1, dtype=th.int64), th.cumsum(n_atoms, dim=0)])
    out[Props.mol_idx] = th.repeat_interleave(
        th.arange(len(batch)), repeats=n_atoms, dim=0
    )

    for prop in props:
        if property_type[prop] == PropsType.edge_wise:
            out_prop = edge_props_to_local_map[prop]
            out[out_prop] = th.cat(
                [sample[prop] for sample in batch], dim=0
            )  # create local edges
            out[prop] = th.cat(
                [d[prop] + off for d, off in zip(batch, at_cumsum)],
                dim=0,  # create offset global edges
            )
        else:
            out[prop] = th.cat([sample[prop] for sample in batch], dim=0)

    return out


def torchyfy(sample, keys_to_props_map: dict[Props, str]):
    torch_sample = {}
    for prop, key in keys_to_props_map.items():
        val = (
            sample[key].to(property_dtype[prop])
            if isinstance(sample[key], th.Tensor)
            else th.tensor(sample[key], dtype=property_dtype[prop])
        )
        if val.ndim == 0:  # scalars
            val = val.unsqueeze(0)
        torch_sample[prop] = val
    return torch_sample


def collate_fn(
    batch,
    props: list[Props],
    device=None,
    batch_func=batch_tall,
    pre_batch_preprocessors=[],
    post_batch_preprocessors=[],
):
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    batch = [torchyfy(sample, props) for sample in batch]
    n_atoms = th.tensor([len(sample[Props.atomic_numbers]) for sample in batch])
    props = list(props.keys())
    for func in pre_batch_preprocessors:
        batch, new_props = func(batch)
        props += new_props

    out = batch_func(batch, props, n_atoms)
    out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    for func in post_batch_preprocessors:
        out = func(out)
    return out
