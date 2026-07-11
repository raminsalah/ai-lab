import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "01-autograd-engine")
)

from value import Value
from neuron import Layer, Neuron


def main():
    neuron = Neuron(3)
    layer = Layer(3, 2)

    print(neuron)
    print(layer)

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    outputs = layer(x)

    print("Outputs:")

    for output in outputs:
        print(output)

    print("Parameter count:", len(layer.parameters()))

    try:
        neuron([Value(1.0)])
    except ValueError as error:
        print("Validation works:", error)


if __name__ == "__main__":
    main()