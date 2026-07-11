import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "01-autograd-engine")
)

from value import Value
from neuron import Layer


# --------------------------------------------------
# Toy example:
# 2 inputs -> 1 neuron -> 1 output
# --------------------------------------------------

layer = Layer(nin=2, nout=1)

x = [
    Value(2.0),
    Value(-1.0),
]

target = Value(1.0)

print("Initial parameters:")

for i, parameter in enumerate(layer.parameters(), start=1):
    print(f"Parameter {i}: {parameter}")


# Forward pass
output = layer(x)[0]

# Mean squared error for one prediction
loss = (output - target) ** 2

print("\nBefore backward:")
print("Output:", output)
print("Target:", target)
print("Loss:", loss)


# Backward pass
layer.zero_grad()
loss.backward()

print("\nAfter backward:")

for i, parameter in enumerate(layer.parameters(), start=1):
    print(f"Parameter {i}: {parameter}")


# One manual gradient-descent update
learning_rate = 0.1

for parameter in layer.parameters():
    parameter.data -= learning_rate * parameter.grad


# Recalculate output and loss after the update
new_output = layer(x)[0]
new_loss = (new_output - target) ** 2

print("\nAfter one parameter update:")
print("New output:", new_output)
print("New loss:", new_loss)