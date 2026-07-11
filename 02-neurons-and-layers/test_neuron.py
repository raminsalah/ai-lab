from neuron import Layer

layer = Layer(3, 4)

print("Number of neurons:", len(layer.neurons))

for i, neuron in enumerate(layer.neurons):
    print(f"Neuron {i}:")
    print("  weights:", neuron.w)
    print("  bias:", neuron.b)