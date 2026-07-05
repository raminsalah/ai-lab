from neuron import Neuron
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "01-autograd-engine"))

from value import Value

n = Neuron(3)

x = [Value(2.0), Value(3.0), Value(-1.0)]

out = n(x)

print("output =", out)
print("weights =", n.w)
print("bias =", n.b)