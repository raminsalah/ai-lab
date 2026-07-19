import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "01-autograd-engine"))

from value import Value
from mlp import MLP


def main() -> None:
    model = MLP(
        nin=3,
        hidden_sizes=[4, 4],
        nout=1,
        hidden_activation="relu",
    )

    x = [
        Value(2.0),
        Value(-1.0),
        Value(0.5),
    ]

    outputs = model(x)

    print("Input:")
    for index, value in enumerate(x, start=1):
        print(f"x{index}: {value}")

    print("\nModel structure:")
    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}: {layer}")

    print("\nFinal outputs:")
    for index, output in enumerate(outputs, start=1):
        print(f"Output {index}: {output}")

    print("\nNumber of outputs:", len(outputs))

    assert len(outputs) == 1


if __name__ == "__main__":
    main()