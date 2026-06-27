from value import Value

targets = [Value(10.0), Value(5.0), Value(-2.0)]
predictions = [Value(8.0), Value(7.0), Value(0.0)]

learning_rate = 0.1

for step in range(15):
    # reset gradients
    for p in predictions:
        p.grad = 0.0

    # MSE = average of squared errors
    losses = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    total_loss = sum(losses, Value(0.0))
    mean_loss = total_loss / Value(len(losses))

    mean_loss.backward()

    for p in predictions:
        p.data -= learning_rate * p.grad

    print(
        f"step {step}: "
        f"loss={mean_loss.data:.4f}, "
        f"predictions={[round(p.data, 4) for p in predictions]}"
    )