import random
import sys
from pathlib import Path

# Make Phase 1 available
sys.path.append(str(Path(__file__).resolve().parents[1] / "01-autograd-engine"))

from value import Value


class Neuron:

    def __init__(self, nin):

        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        self.b = Value(random.uniform(-1, 1))