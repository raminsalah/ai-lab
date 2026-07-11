import random
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "01-autograd-engine")
)

from value import Value
from module import Module


class Neuron(Module):
    """A single fully connected neuron with ReLU activation."""

    def __init__(self, nin: int):
        super().__init__()

        if nin <= 0:
            raise ValueError("nin must be greater than zero")

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

        activation = sum(
            (weight * value for weight, value in zip(self.w, x)),
            self.b,
        )

        return activation.relu()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron(nin={len(self.w)}, activation=ReLU)"


class Layer(Module):
    """A fully connected layer containing multiple neurons."""

    def __init__(self, nin: int, nout: int):
        super().__init__()

        if nin <= 0:
            raise ValueError("nin must be greater than zero")

        if nout <= 0:
            raise ValueError("nout must be greater than zero")

        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, x: Iterable[Value]) -> List[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> List[Value]:
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]

    def train(self):
        super().train()

        for neuron in self.neurons:
            neuron.train()

        return self

    def eval(self):
        super().eval()

        for neuron in self.neurons:
            neuron.eval()

        return self

    def __repr__(self) -> str:
        return (
            f"Layer(nin={len(self.neurons[0].w)}, "
            f"nout={len(self.neurons)})"
        )