from mlp import MLP


def main():
    model = MLP()

    print("Number of layers:", len(model.layers))

    print("\nLayers:")

    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}: {layer}")

    print("\nTotal parameters:", len(model.parameters()))

    print("\nInitial training mode:")
    print("MLP:", model.training)

    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}:", layer.training)

    model.eval()

    print("\nAfter eval():")
    print("MLP:", model.training)

    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}:", layer.training)

    model.train()

    print("\nAfter train():")
    print("MLP:", model.training)

    for index, layer in enumerate(model.layers, start=1):
        print(f"Layer {index}:", layer.training)


if __name__ == "__main__":
    main()