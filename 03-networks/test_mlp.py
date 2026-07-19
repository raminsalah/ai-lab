from mlp import MLP


def print_model_structure(name: str, model: MLP) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print("Input size:", model.nin)
    print("Hidden sizes:", model.hidden_sizes)
    print("Output size:", model.nout)
    print("Hidden activation:", model.hidden_activation)
    print("Number of layers:", len(model.layers))

    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}: {layer}")

    print("Total parameters:", len(model.parameters()))


def main() -> None:
    relu_model = MLP(
        nin=3,
        hidden_sizes=[4, 4],
        nout=1,
        hidden_activation="relu",
    )

    tanh_model = MLP(
        nin=2,
        hidden_sizes=[3, 3],
        nout=1,
        hidden_activation="tanh",
    )

    direct_model = MLP(
        nin=2,
        hidden_sizes=[],
        nout=1,
    )

    print_model_structure(
        "ReLU model: 3 → 4 → 4 → 1",
        relu_model,
    )

    print_model_structure(
        "Tanh model: 2 → 3 → 3 → 1",
        tanh_model,
    )

    print_model_structure(
        "No hidden layer: 2 → 1",
        direct_model,
    )


if __name__ == "__main__":
    main()