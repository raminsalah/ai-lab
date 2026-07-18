import sys
from pathlib import Path
from typing import List

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "02-neurons-and-layers")
)

from module import Module
from neuron import Layer
from value import Value


class MLP(Module):
    """A simple multi-layer perceptron with a fixed architecture."""

    def __init__(self):
        super().__init__()

        self.layers = [
            Layer(nin=3, nout=4),
            Layer(nin=4, nout=4),
            Layer(nin=4, nout=1),
        ]

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