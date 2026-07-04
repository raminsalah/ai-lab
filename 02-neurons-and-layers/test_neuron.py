from neuron import Neuron

n = Neuron(3)

print("Weights:")

for w in n.w:
    print(w)

print()

print("Bias:")

print(n.b)