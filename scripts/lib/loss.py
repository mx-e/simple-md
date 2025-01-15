from enum import Enum

import torch as th
import torch.nn.functional as F
from lib.types import Property as Props
from lib.types import PropertyType, property_type
from torch import nn


def atom_wise_euclidean(pred_forces, true_forces, reduction="mean") -> th.Tensor:
    assert reduction in ["mean", "none"], f"Invalid reduction {reduction}"
    diff = pred_forces - true_forces  # (n, 3)
    diff = diff.norm(dim=-1)  # (n,)
    if reduction == "none":
        return diff
    return diff.mean()

def atom_wise_cosine_similarity(pred_forces, true_forces, reduction="mean") -> th.Tensor:
    assert reduction in ["mean", "none"], f"Invalid reduction {reduction}"
    pred_forces = F.normalize(pred_forces, p=2, dim=-1)
    true_forces = F.normalize(true_forces, p=2, dim=-1)
    sim = (pred_forces * true_forces).sum(dim=-1)  # (n,)
    if reduction == "none":
        return sim
    return sim.mean()

class LossType(Enum):
    force_weighted = "force_weighted"
    mae = "mae"
    euclidean = "euclidean"
    mse = "mse"
    huber = "huber"
    rve = "rve"
    rmse = "rmse"
    cosine = "cosine"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value) -> "LossType":
        # This allows Property["energy"] to work the same as Property.energy
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


def rmse(pred, true, reduction="mean") -> th.Tensor:
    return th.sqrt(F.mse_loss(pred, true, reduction=reduction))


force_loss_funcs = {
    LossType.mae: F.l1_loss,
    LossType.rmse: rmse,
    LossType.euclidean: atom_wise_euclidean,
    LossType.mse: F.mse_loss,
    LossType.huber: F.smooth_l1_loss,
    LossType.cosine: atom_wise_cosine_similarity,
}

energy_loss_funcs = {
    LossType.mae: F.l1_loss,
    LossType.mse: F.mse_loss,
    LossType.huber: F.smooth_l1_loss,
    LossType.rmse: rmse,
}

dipole_loss_funcs = {
    LossType.mae: F.l1_loss,
    LossType.euclidean: atom_wise_euclidean,
    LossType.mse: F.mse_loss,
    LossType.huber: F.smooth_l1_loss,
    LossType.rmse: rmse,
}

loss_funcs = {
    Props.forces: force_loss_funcs,
    Props.formation_energy: energy_loss_funcs,
    Props.energy: energy_loss_funcs,
    Props.dipole: dipole_loss_funcs,
}

atomref_val_targets = {
    Props.formation_energy: Props.formation_energy_atomref,
    Props.energy: Props.energy_atomref,
}


def unbatch(batched_tensor, mask) -> th.Tensor:
    mask = mask.unsqueeze(-1).expand_as(batched_tensor)  # (b, n, 3)
    unbatched_tensor = batched_tensor.masked_select(mask).view(-1, batched_tensor.shape[-1])  # (n', 3)
    return unbatched_tensor


def mean_losses_elementwise(unbatched_losses, mask) -> th.Tensor:
    batch_size = mask.shape[0]  # (b, n)
    n_dims = len(unbatched_losses.shape)
    indices = th.repeat_interleave(
        th.arange(batch_size, device=unbatched_losses.device),
        mask.sum(dim=1),  # (b,)
    )  # (n',)
    indices = indices.unsqueeze(-1) if n_dims == 2 else indices
    per_mol_loss = th.zeros((batch_size, 1) if n_dims == 2 else batch_size, device=unbatched_losses.device)  # (b,)
    per_mol_loss.scatter_add_(dim=0, index=indices, src=unbatched_losses).squeeze()  # (b,)
    return per_mol_loss / mask.sum(dim=1)  # (b,)


class LossModule(nn.Module):
    def __init__(
        self,
        targets: list[str],
        loss_types: dict[str, str],
        metrics: dict[str, list],
        weights: dict[str, float] | None = None,
        losses_per_mol: bool = False,
        compute_metrics_train: bool = False,
    ) -> None:
        super().__init__()
        # resolve types
        self.targets = [Props[target] for target in targets]
        self.weights = (
            {Props[target]: w for target, w in weights.items()} if weights else {t: 1.0 for t in self.targets}
        )
        self.loss_types = {Props[target]: LossType[lt] for target, lt in loss_types.items()}
        self.metrics = {Props[target]: [LossType[m] for m in metric_list] for target, metric_list in metrics.items()}

        self.losses_per_mol = losses_per_mol
        self.compute_metrics_train = compute_metrics_train
        assert (
            len(self.targets) == len(self.weights) == len(loss_types)
        ), "For each target, a loss weight and a loss type must be configured"

        try:
            self.loss_funcs = {target: {lt: loss_funcs[target][lt]} for target, lt in self.loss_types.items()}
            self.metric_funcs = {
                target: {metric: loss_funcs[target][metric] for metric in metric_list}
                for target, metric_list in self.metrics.items()
            }
        except KeyError as e:
            raise ValueError(f"Metric function not defined for one or more targets: {e}") from e
        self.loss_funcs_val = {target: self.loss_funcs[target] | self.metric_funcs[target] for target in self.targets}

    def forward(self, predictions, inputs) -> dict[str | Props, th.Tensor]:
        losses = {}
        for loss_prop in self.targets:
            loss_funcs = (
                self.loss_funcs_val[loss_prop]
                if not self.training or self.compute_metrics_train
                else self.loss_funcs[loss_prop]
            )
            for loss_type, loss_func in loss_funcs.items():
                loss_dict_key = f"{loss_prop!s}_{loss_type!s}"
                losses[loss_dict_key] = self.compute_metric(predictions, inputs, loss_prop, loss_func)
        if self.losses_per_mol:
            return losses
        weighted_train_losses = {
            target: losses[f"{target!s}_{loss_func!s}"] * self.weights[target]
            for target, loss_func in self.loss_types.items()
        }
        losses["total"] = sum(weighted_train_losses.values())
        return losses

    def compute_metric(self, predictions, inputs, loss_prop, loss_func) -> th.Tensor:
        loss_comparison_prop = loss_prop
        # handle atomref targets
        if loss_prop in atomref_val_targets and not self.training:
            loss_comparison_prop = atomref_val_targets[loss_prop]
        reduction = "none" if self.losses_per_mol and not self.training else "mean"

        if property_type[loss_prop] == PropertyType.mol_wise:
            return loss_func(predictions[loss_comparison_prop], inputs[loss_comparison_prop], reduction=reduction)
        else:
            # unbatch atom wise properties and remove padding
            pred = predictions[loss_comparison_prop]
            target = inputs[loss_comparison_prop]
            is_batched = Props.mask in inputs
            if is_batched:
                pred = unbatch(pred, inputs[Props.mask])
                target = unbatch(target, inputs[Props.mask])
            loss = loss_func(pred, target, reduction=reduction)
            # mean losses per molecule if requested
            if not self.losses_per_mol:
                return loss
            return mean_losses_elementwise(loss, inputs[Props.mask])
