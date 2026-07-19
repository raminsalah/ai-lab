import random
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "01-autograd-engine"))

from value import Value
from module import Module


class Neuron(Module):
    """A fully connected neuron with a configurable activation."""

    SUPPORTED_ACTIVATIONS = {"relu", "tanh", "linear"}

    def __init__(self, nin: int, activation: str = "relu"):
        super().__init__()

        if nin <= 0:
            raise ValueError("nin must be greater than zero")

        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation!r}. "
                f"Choose from {sorted(self.SUPPORTED_ACTIVATIONS)}"
            )

        self.activation = activation

        self.w = [
            Value(random.uniform(-1.0, 1.0))
            for _ in range(nin)
        ]

        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x: Iterable[Value]) -> Value:
        x = list(x)

        if len(x) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} inputs, got {len(x)}"
            )

        output = sum(
            (
                weight * input_value
                for weight, input_value in zip(self.w, x)
            ),
            self.b,
        )

        if self.activation == "relu":
            return output.relu()

        if self.activation == "tanh":
            return output.tanh()

        return output

    def parameters(self) -> List[Value]:
        """Return this neuron's weights and bias."""
        return self.w + [self.b]

    def __repr__(self) -> str:
        return (
            f"Neuron("
            f"nin={len(self.w)}, "
            f"activation={self.activation}"
            f")"
        )


class Layer(Module):
    """A fully connected layer containing multiple neurons."""

    def __init__(
        self,
        nin: int,
        nout: int,
        activation: str = "relu",
    ):
        super().__init__()

        if nin <= 0:
            raise ValueError("nin must be greater than zero")

        if nout <= 0:
            raise ValueError("nout must be greater than zero")

        self.nin = nin
        self.nout = nout
        self.activation = activation

        self.neurons = [
            Neuron(
                nin=nin,
                activation=activation,
            )
            for _ in range(nout)
        ]

    def __call__(self, x: Iterable[Value]) -> List[Value]:
        """Pass the same input vector through every neuron."""
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> List[Value]:
        """Return all parameters from all neurons."""
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]

    def train(self):
        """Switch the layer and its neurons to training mode."""
        super().train()

        for neuron in self.neurons:
            neuron.train()

        return self

    def eval(self):
        """Switch the layer and its neurons to evaluation mode."""
        super().eval()

        for neuron in self.neurons:
            neuron.eval()

        return self

    def __repr__(self) -> str:
        return (
            f"Layer("
            f"nin={self.nin}, "
            f"nout={self.nout}, "
            f"activation={self.activation}"
            f")"
        )