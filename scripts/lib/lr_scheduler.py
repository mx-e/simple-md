import math

import torch as th
from loguru import logger


class LRScheduler:
    def __init__(self, optimizer, lr) -> None:
        self.optimizer = optimizer
        self.lr = optimizer.param_groups[0]["lr"]
        self.initial_lr = lr

    def step(self, step: int) -> None:
        pass

    def step_on_loss(self, step, loss: float) -> None:
        pass

    def set_lr(self, lr: float) -> None:
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def get_lr_scheduler(optimizer, lr, scheduler_type: str, **kwargs) -> LRScheduler:
    match scheduler_type:
        case "piecewise_linear":
            return PieceWiseLinearLRScheduler(optimizer, lr, **kwargs)
        case "adaptive_anneal":
            return AdaptiveAnnealingScheduler(optimizer, lr, **kwargs)
        case "cosine_warmup":
            return CosineWithWarmupLRScheduler(optimizer, lr, **kwargs)
        case _:
            logger.warning(
                f"No LR scheduler configured - running with constant LR ({optimizer.param_groups[0]['lr']})"
            )
            return LRScheduler(optimizer, **kwargs)


class PieceWiseLinearLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr, milestones: list[tuple[int, float]]) -> None:
        super().__init__(optimizer, lr)
        ## milestone in the format of (step, lr)
        milestones = sorted(milestones, key=lambda x: x[0])
        self.ms_steps = th.tensor([x[0] for x in milestones])
        self.ms_lr = th.tensor([x[1] for x in milestones])
        assert (self.ms_steps[1:] - self.ms_steps[:-1]).min() > 0, "Lr steps must be unique and ascending"

    def step(self, step: int) -> None:
        if step <= self.ms_steps[0].item():
            lr = self.ms_lr[0].item()
        elif step >= self.ms_steps[-1].item():
            lr = self.ms_lr[-1].item()
        else:
            idx = th.searchsorted(self.ms_steps, step).item()
            left_step, right_step = self.ms_steps[idx - 1], self.ms_steps[idx]
            left_lr, right_lr = self.ms_lr[idx - 1], self.ms_lr[idx]

            lr = left_lr + (right_lr - left_lr) * (step - left_step) / (right_step - left_step)
            lr = lr.item()

        self.set_lr(lr)


class CosineWithWarmupLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        lr,
        warmup_steps: int,
        lr_decay_steps: int,
        min_lr: float,
    ) -> None:
        super().__init__(optimizer, lr)
        self.warmup_iters = warmup_steps
        self.lr_decay_iters = lr_decay_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self, step: int) -> None:
        self.current_step = step
        lr = self._get_lr(step)
        self.set_lr(lr)

    def _get_lr(self, step: int) -> float:
        lr = self.initial_lr
        if step < self.warmup_iters:
            return lr * step / self.warmup_iters
        if step > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (step - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (lr - self.min_lr)


class AdaptiveAnnealingScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        anneal_after_sideways_steps,
        anneal_factor,
        min_lr,
        threshold=0.0,
    ) -> None:
        super().__init__(optimizer)
        self.anneal_after_sideways_steps = anneal_after_sideways_steps
        self.anneal_factor = anneal_factor
        self.current_lr = self.lr
        self.best_energy_loss = th.inf
        self.best_energy_loss_update = 0
        self.best_forces_loss = th.inf
        self.best_forces_loss_update = 0
        self.first_time_called = True
        self.minimal_learning_rate = min_lr
        self.threshold = threshold

    def step_on_loss(self, step, loss: float) -> None:
        # when called for the first time, just set the best losses to the current loss
        # important when loading a model and continuing training
        energy_loss = loss["energy_loss"]
        forces_loss = loss["forces_loss"]
        if self.first_time_called:
            self.first_time_called = False
            self.best_energy_loss = energy_loss
            self.best_energy_loss_update = step
            self.best_forces_loss = forces_loss
            self.best_forces_loss_update = step
            return

        if energy_loss < self.best_energy_loss * (1 - self.threshold) or (
            forces_loss < self.best_forces_loss * (1 - self.threshold)
        ):
            if energy_loss < self.best_energy_loss:
                self.best_energy_loss = energy_loss
                self.best_energy_loss_update = step
                logger.info(f"Energy loss improved to {energy_loss}")
            if forces_loss < self.best_forces_loss:
                self.best_forces_loss = forces_loss
                self.best_forces_loss_update = step
                logger.info(f"Forces loss improved to {forces_loss}")
        # anneal if no improvement in either energy or forces
        elif (
            min(step - self.best_energy_loss_update, step - self.best_forces_loss_update)
            >= self.anneal_after_sideways_steps
        ):
            if self.current_lr * self.anneal_factor < self.minimal_learning_rate:
                logger.info("Minimal learning rate reached, not annealing.")
                return
            if step - self.best_energy_loss_update >= self.anneal_after_sideways_steps:
                logger.info(f"No improvement in energy loss for {step - self.best_energy_loss_update} steps.")
            if step - self.best_forces_loss_update >= self.anneal_after_sideways_steps:
                logger.info(f"No improvement in forces loss for {step - self.best_forces_loss_update} steps.")
            logger.info(f"Annealing LR from {self.current_lr} to {self.current_lr * self.anneal_factor}")
            self.current_lr *= self.anneal_factor
            self.set_lr(self.current_lr)
            self.best_energy_loss = min(self.best_energy_loss, energy_loss)
            self.best_energy_loss_update = step
            self.best_forces_loss = min(self.best_forces_loss, forces_loss)
            self.best_forces_loss_update = step
