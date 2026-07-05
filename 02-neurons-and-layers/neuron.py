import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "01-autograd-engine"))

from value import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu()
        return out