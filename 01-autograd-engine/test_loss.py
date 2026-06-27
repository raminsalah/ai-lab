from value import Value
from losses import mse_loss

targets = [Value(10.0), Value(5.0), Value(-2.0)]
predictions = [Value(8.0), Value(7.0), Value(0.0)]

learning_rate = 0.1

for step in range(10):
    for p in predictions:
        p.grad = 0.0

    loss = mse_loss(predictions, targets)
    loss.backward()

    for p in predictions:
        p.data -= learning_rate * p.grad

    print(
        f"step {step}: "
        f"loss={loss.data:.4f}, "
        f"predictions={[round(p.data, 4) for p in predictions]}"
    )