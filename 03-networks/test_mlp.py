from mlp import MLP


def print_model_structure(name: str, model: MLP) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print("Input size:", model.nin)
    print("Hidden sizes:", model.hidden_sizes)
    print("Output size:", model.nout)
    print("Number of layers:", len(model.layers))

    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}: {layer}")

    print("Total parameters:", len(model.parameters()))


def main() -> None:
    small_model = MLP(
        nin=3,
        hidden_sizes=[4, 4],
        nout=1,
    )

    wider_model = MLP(
        nin=5,
        hidden_sizes=[8, 4, 2],
        nout=3,
    )

    direct_model = MLP(
        nin=2,
        hidden_sizes=[],
        nout=1,
    )

    print_model_structure(
        "Small model: 3 → 4 → 4 → 1",
        small_model,
    )

    print_model_structure(
        "Wider model: 5 → 8 → 4 → 2 → 3",
        wider_model,
    )

    print_model_structure(
        "No hidden layer: 2 → 1",
        direct_model,
    )


if __name__ == "__main__":
    main()