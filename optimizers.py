import torch


class SGD:
    def __init__(self):
        pass

    def register_params(self, parameters: dict[str, torch.Tensor]) -> None:
        self.params = parameters

    def step(self, lr) -> None:
        for k, param in self.params.items():
            param.data += -lr * param.grad

    def zero_grad(self) -> None:
        for k, param in self.params.items():
            param.grad = None
