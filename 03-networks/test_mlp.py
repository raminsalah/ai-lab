import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "01-autograd-engine"))

from value import Value
from mlp import MLP


def print_parameter_breakdown(model: MLP) -> None:
    print("\nParameter breakdown:")

    total_from_layers = 0

    for index, layer in enumerate(model.layers, start=1):
        layer_parameter_count = len(layer.parameters())
        total_from_layers += layer_parameter_count

        print(
            f"Layer {index}: "
            f"{layer.nin} → {layer.nout}, "
            f"parameters={layer_parameter_count}"
        )

    print("Total from layers:", total_from_layers)
    print("Total from model :", model.count_parameters())


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

    print("Model architecture:")
    print("Input size:", model.nin)
    print("Hidden sizes:", model.hidden_sizes)
    print("Output size:", model.nout)

    print("\nFinal output:")
    print(outputs[0])

    print_parameter_breakdown(model)

    assert model.count_parameters() == 41
    assert model.count_parameters() == len(model.parameters())


if __name__ == "__main__":
    main()