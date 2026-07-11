# Phase 1.8 – Relationship Between Activation Functions, Loss Functions, Backpropagation, Weights and Optimizer

---

# Big Picture

Training an AI model is simply repeating the following cycle:

```text

                Input Data
                     │
                     ▼
         ┌─────────────────────┐
         │      Weights        │
         │  (learnable values) │
         └─────────────────────┘
                     │
                     ▼
              Linear Equation
          z = w·x + b (Weighted Sum)

                     │
                     ▼
            Activation Function
      (ReLU / GELU / Tanh / Sigmoid)

                     │
                     ▼
               Model Prediction

                     │
                     ▼
            Compare with Reality
            (Ground Truth Label)

                     │
                     ▼
               Loss Function
             "How wrong was I?"

                     │
                     ▼
              Backpropagation
          "Who caused this error?"

                     │
                     ▼
                 Gradients
               ∂Loss/∂Weight

                     │
                     ▼
                 Optimizer
     "How should I change the weights?"

                     │
                     ▼
             Updated Weights

                     │
                     └───────────────► Repeat thousands of times
```

---

# Step 1 — Input

Suppose we have a very small neural network.

Input:

```text
House size = 100 m²
```

---

# Step 2 — Weight

Initially the network has

```text
Weight = 0.5
Bias   = 2
```

These numbers are completely random at first.

The network knows nothing.

---

# Step 3 — Linear Equation

Every neuron first computes

```
z = w·x + b
```

Example

```
z = 0.5 × 100 + 2

z = 52
```

This is called the **weighted sum**.

---

# Step 4 — Activation Function (ReLu-GELU)

Suppose we use ReLU.

```
ReLU(52)

↓

52
```

Nothing changes because it is positive.

If instead

```
z = -7
```

then

```
ReLU(-7)

↓

0
```

The neuron becomes inactive.

Activation Functions decide

> Should this neuron pass information forward?

Examples

```
ReLU

Negative → OFF
Positive → ON
```

```
GELU (Gaussian Error Linear Unit)

Soft ON/OFF

Negative values are only partially suppressed.
```

---

# Step 5 — Prediction

The network predicts

```
Predicted price = €520,000
```

Reality is

```
True price = €500,000
```

Difference

```
20,000
```

---

# Step 6 — Loss Function (like MSE)

Loss answers

> How wrong was I?

Using Mean Squared Error (MSE)

```
Loss

=

(Prediction − Truth)²

=

(520−500)²

=

20²

=

400
```

The network now knows

"I made an error of 400."

---

# Step 7 — Backpropagation

Now we ask

> Which weights caused this error?

Backpropagation walks backward

```
Loss

▲

Prediction

▲

Activation

▲

Weights
```

and computes

```
∂Loss/∂Weight
```

Suppose

```
Weight gradient = +8
```

Interpretation

Increasing this weight increases the loss.

So the optimizer should decrease it.

---

# Step 8 — Optimizer

The optimizer updates the weights.

Most common:

Gradient Descent

Formula

```
new_weight = old_weight − (learning_rate × gradient)
```

Example

Old weight

```
0.5
```

Gradient

```
8
```

Learning rate

```
0.01
```

Update

```
new_weight = 0.5 − (0.01×8) = 0.42
```

The weight has moved toward a better value.

---

# Step 9 — Repeat

The network performs

```
Input

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Weight Update
```

millions of times.

Eventually

```
Prediction

↓

Truth
```

and

```
Loss

↓

Almost Zero
```

---

# Relationship Between Components

| Component | Purpose |
|------------|---------|
| Weight | Stores learned knowledge |
| Linear equation | Combines inputs |
| Activation Function | Decides how much information continues |
| Prediction | Current answer |
| Loss Function | Measures how wrong the prediction is |
| Backpropagation | Finds which parameters caused the error |
| Gradient | Indicates the direction and magnitude of change |
| Optimizer | Updates the weights |
| Training Loop | Repeats the entire process |

---

# One Sentence Summary

Weights make predictions.

↓

Activation functions decide which neurons activate.

↓

Loss function measures how wrong the prediction is.

↓

Backpropagation computes how every weight contributed to that error.

↓

Optimizer adjusts the weights.

↓

Repeat until the loss becomes very small.

This cycle is the heart of every modern AI model, from a tiny neural network to GPT.