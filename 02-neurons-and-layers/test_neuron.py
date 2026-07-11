from neuron import Layer


layer = Layer(3, 4)

print("Initial mode:")
print("Layer training:", layer.training)

for i, neuron in enumerate(layer.neurons, start=1):
    print(f"Neuron {i} training:", neuron.training)


print("\nSwitching to eval mode...")

layer.eval()

print("Layer training:", layer.training)

for i, neuron in enumerate(layer.neurons, start=1):
    print(f"Neuron {i} training:", neuron.training)


print("\nSwitching back to train mode...")

layer.train()

print("Layer training:", layer.training)

for i, neuron in enumerate(layer.neurons, start=1):
    print(f"Neuron {i} training:", neuron.training)