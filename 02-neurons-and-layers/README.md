# Phase 2 — Neurons and Layers

Phase 2 builds neural-network components on top of the scalar autograd engine created in Phase 1.

The purpose of this phase is to understand how trainable weights, biases, neurons, and layers are represented internally before building a complete multilayer neural network.

## Learning objectives

By completing this phase, I learned how to:

* create trainable weights and biases using `Value`
* implement a single artificial neuron
* perform a neuron forward pass
* combine multiple neurons into a layer
* collect trainable parameters
* reset accumulated gradients
* switch components between training and evaluation modes
* test forward propagation, backpropagation, and parameter updates
* organize the code into reusable neural-network modules

## Roadmap

* [x] 2.1 Neuron Class
* [x] 2.2 Forward Pass
* [x] 2.3 Layer Class
* [x] 2.4 Forward Pass for Layer
* [x] 2.5 `parameters()` and `zero_grad()`
* [x] 2.6 Train/Eval Mode
* [x] 2.7 Testing with Toy Examples
* [x] 2.8 Refactor and Clean Code
* [x] 2.9 Documentation

## Artificial neuron

A neuron receives multiple inputs, multiplies each input by a trainable weight, adds a bias, and applies an activation function.

The neuron computes:

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
output = ReLU(z)
```

Where:

* `x1 ... xn` are input values
* `w1 ... wn` are trainable weights
* `b` is a trainable bias
* `ReLU` is the activation function

## Data flow

```text
Inputs
  │
  ├── x1 × w1
  ├── x2 × w2
  └── xn × wn
        │
        ▼
Weighted sum + bias
        │
        ▼
      ReLU
        │
        ▼
      Output
```

## Neuron example

```python
neuron = Neuron(3)

x = [
    Value(2.0),
    Value(3.0),
    Value(-1.0),
]

output = neuron(x)
```

`Neuron(3)` creates:

```text
3 weights
1 bias
```

The output remains a `Value` object, so the computational graph and gradients are preserved.

## Layer

A layer contains multiple neurons.

```python
layer = Layer(nin=3, nout=4)
```

This means:

```text
3 inputs
4 neurons
4 outputs
```

Every neuron receives the same input values but has its own weights and bias.

```text
Input vector
   │
   ├── Neuron 1 → Output 1
   ├── Neuron 2 → Output 2
   ├── Neuron 3 → Output 3
   └── Neuron 4 → Output 4
```

## Parameters

The `parameters()` method returns every trainable weight and bias.

For:

```python
layer = Layer(3, 4)
```

each neuron contains:

```text
3 weights + 1 bias = 4 parameters
```

The layer therefore contains:

```text
4 neurons × 4 parameters = 16 parameters
```

## Gradient reset

The autograd engine accumulates gradients using:

```python
parameter.grad += ...
```

Therefore gradients must normally be cleared before each new backward pass:

```python
layer.zero_grad()
```

A typical training step follows this order:

```text
zero gradients
→ forward pass
→ compute loss
→ backward pass
→ update parameters
```

## Train and evaluation modes

All modules start in training mode:

```python
module.training is True
```

The mode can be changed using:

```python
layer.train()
layer.eval()
```

The current neuron and layer calculations behave the same in both modes.

The mode system is included because later components such as Dropout and Batch Normalization behave differently during training and evaluation.

## Toy training example

A single-neuron layer can perform the full learning cycle:

```python
output = layer(x)[0]
loss = (output - target) ** 2

layer.zero_grad()
loss.backward()

for parameter in layer.parameters():
    parameter.data -= learning_rate * parameter.grad
```

The complete flow is:

```text
Input
  ↓
Neuron
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Parameter gradients
  ↓
Parameter update
```

## Files

```text
02-neurons-and-layers/
├── __init__.py
├── module.py
├── neuron.py
├── test_neuron.py
├── test_toy_examples.py
├── test_refactor.py
└── README.md
```

## Main classes

### `Module`

Provides:

* `parameters()`
* `zero_grad()`
* `train()`
* `eval()`

### `Neuron`

Provides:

* random weight initialization
* trainable bias
* forward propagation
* ReLU activation
* parameter collection

### `Layer`

Provides:

* multiple neurons
* multiple outputs
* flattened parameter collection
* train/eval mode propagation

## Phase result

At the end of Phase 2, the project contains reusable neuron and layer components that:

* use the custom Phase 1 autograd engine
* preserve computational graphs
* support backward propagation
* expose trainable parameters
* support parameter updates
* provide a foundation for multilayer neural networks

The next phase will combine multiple layers into a Multi-Layer Perceptron.
