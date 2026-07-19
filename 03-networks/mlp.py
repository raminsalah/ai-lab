import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "01-autograd-engine"))
sys.path.append(str(PROJECT_ROOT / "02-neurons-and-layers"))

from value import Value
from module import Module
from neuron import Layer


class MLP(Module):
    """A multi-layer perceptron with configurable hidden activation."""

    def __init__(
        self,
        nin: int,
        hidden_sizes: List[int],
        nout: int,
        hidden_activation: str = "relu",
    ):
        super().__init__()

        if nin <= 0:
            raise ValueError("nin must be greater than zero")

        if nout <= 0:
            raise ValueError("nout must be greater than zero")

        if any(size <= 0 for size in hidden_sizes):
            raise ValueError(
                "All hidden layer sizes must be greater than zero"
            )

        self.nin = nin
        self.hidden_sizes = list(hidden_sizes)
        self.nout = nout
        self.hidden_activation = hidden_activation

        layer_sizes = [
            nin,
            *hidden_sizes,
            nout,
        ]

        self.layers = []

        for index in range(len(layer_sizes) - 1):
            is_output_layer = index == len(layer_sizes) - 2

            activation = (
                "linear"
                if is_output_layer
                else hidden_activation
            )

            layer = Layer(
                nin=layer_sizes[index],
                nout=layer_sizes[index + 1],
                activation=activation,
            )

            self.layers.append(layer)

    def __call__(self, x: Iterable[Value]) -> List[Value]:
        """Pass an input vector through every layer in sequence."""
        outputs = list(x)

        if len(outputs) != self.nin:
            raise ValueError(
                f"Expected {self.nin} inputs, got {len(outputs)}"
            )

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

    def parameters(self) -> List[Value]:
        """Return all trainable parameters from all layers."""
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]

    def train(self):
        """Switch the MLP and all child layers to training mode."""
        super().train()

        for layer in self.layers:
            layer.train()

        return self

    def eval(self):
        """Switch the MLP and all child layers to evaluation mode."""
        super().eval()

        for layer in self.layers:
            layer.eval()

        return self