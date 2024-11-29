import torch as th


def save_checkpoint(model, optimizer, step, path, ema=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    if ema is not None:
        ckpt["ema_state"] = ema.state_dict()
    th.save(ckpt, path)


def load_checkpoint(
    model, path, optimizer=None, ema=None, step_back=200
):  # stepping back improves stability when using torch compile and mutliple GPUs (its unstable when starting directly with eval)
    device = next(model.parameters()).device
    checkpoint = th.load(path, weights_only=True, map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if ema is not None and "ema_state" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"] - step_back
