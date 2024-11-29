import torch as th
import torch.nn.functional as F
from torch import nn
from lib.types import Property as Props
from enum import Enum


def atom_weighted_forces_mae(pred_forces, true_forces, mask, reduction="mean"):
    assert reduction in ["mean", "none"], f"Invalid reduction {reduction}"
    diff = (pred_forces - true_forces) * mask  # (b, n, 3) / (n, 3)
    diff = diff.norm(dim=-1)  # (b, n) / (n,)
    mask = mask.squeeze(-1)  # (b, n) / (n,)
    diff = diff.sum(dim=-1) / mask.sum(dim=1)  # (b,) / ()
    if reduction == "none":
        return diff
    return diff.mean()


def atom_weighted_forces_mse(pred_forces, true_forces, mask, reduction="mean"):
    assert reduction in ["mean", "none"], f"Invalid reduction {reduction}"
    diff = ((pred_forces - true_forces) ** 2) * mask  # (b, n, 3)
    diff = diff.sum(dim=-1)  # (b, n)
    mask = mask.squeeze(-1)
    diff = diff.sum(dim=1) / mask.sum(dim=1)
    if reduction == "none":
        return diff
    return diff.mean()


def atom_weighted_forces_huber(
    pred_forces, true_forces, mask, reduction="mean", delta=1.0
):
    assert reduction in ["mean", "none"], f"Invalid reduction {reduction}"
    diff = (pred_forces - true_forces) * mask  # (b, n, 3) / (n, 3)
    abs_diff = diff.abs()

    quadratic = th.min(abs_diff, th.full_like(abs_diff, delta))
    linear = abs_diff - quadratic
    losses = 0.5 * quadratic**2 + delta * linear

    losses = losses.sum(dim=-1)  # (b, n)
    mask = mask.squeeze(-1)
    losses = losses.sum(dim=1) / mask.sum(dim=1)
    if reduction == "none":
        return losses
    return losses.mean()


class LossType(Enum):
    force_weighted = "force_weighted"
    mae = "mae"
    mse = "mse"
    huber = "huber"
    rve = "rve"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def _missing_(cls, value):
        # This allows Property["energy"] to work the same as Property.energy
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


force_loss_funcs = {
    LossType.mae: atom_weighted_forces_mae,
    LossType.mse: atom_weighted_forces_mse,
    LossType.huber: atom_weighted_forces_huber,
}

energy_loss_funcs = {
    LossType.mae: F.l1_loss,
    LossType.mse: F.mse_loss,
    LossType.huber: F.smooth_l1_loss,
}

dipole_loss_funcs = {
    LossType.mae: F.l1_loss,
    LossType.mse: F.mse_loss,
    LossType.huber: F.smooth_l1_loss,
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


class LossModule(nn.Module):
    def __init__(
        self,
        targets: list[str],
        loss_types: dict[str, str],
        weights: dict[str, float] | None = None,
        losses_per_atom: bool = False,
    ):
        super().__init__()
        self.targets = [Props[t] for t in targets]
        self.weights = (
            {Props[t]: w for t, w in weights.items()}
            if weights
            else {t: 1.0 for t in self.targets}
        )
        self.loss_types = {Props[t]: LossType[lt] for t, lt in loss_types.items()}
        self.losses_per_atom = losses_per_atom
        assert (
            len(self.targets) == len(self.weights) == len(self.loss_types)
        ), "For each target, a loss weight and a loss type must be configured"

        try:
            self.loss_funcs = {k: loss_funcs[k][v] for k, v in self.loss_types.items()}
            self.loss_funcs_val = {k: loss_funcs[k][LossType.mae] for k in self.targets}
        except KeyError as e:
            raise ValueError(f"Loss function not defined for one or more targets: {e}")

    def forward(self, predictions, inputs):
        if Props.mask not in inputs:  # treat flat batch as a tall batch of size 1
            assert (
                self.energy_loss_type != "force_weighted"
            ), "Force weighted loss not implemented for flat batches"
            mask = th.ones(
                1,
                predictions[Props.forces].shape[0],
                1,
                device=predictions[Props.forces].device,
            )
            if Props.forces in predictions:
                predictions[Props.forces].unsqueeze_(0)
                inputs[Props.forces].unsqueeze_(0)
        else:
            mask = inputs[Props.mask].unsqueeze(-1)

        losses = {}
        for loss_prop in self.targets:
            loss_func = (
                self.loss_funcs[loss_prop]
                if self.training
                else self.loss_funcs_val[loss_prop]
            )
            reduction = "none" if self.losses_per_atom and not self.training else "mean"
            # handle atomref targets
            loss_dict_prop = loss_prop
            if loss_prop in atomref_val_targets and not self.training:
                loss_prop = atomref_val_targets[loss_prop]

            losses[loss_dict_prop] = loss_func(
                predictions[loss_prop], inputs[loss_prop], mask, reduction=reduction
            )

        if self.losses_per_atom:
            return losses
        weighted_losses = {k: v * self.weights[k] for k, v in losses.items()}
        losses["total"] = sum(weighted_losses.values())
        return losses
