class EMAModel:
    def __init__(self, model, decay, device=None) -> None:
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow, "backup": self.backup}

    def load_state_dict(self, state_dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self.backup = state_dict["backup"]
