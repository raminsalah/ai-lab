from value import Value

target = Value(10.0)
prediction = Value(8.0)

learning_rate = 0.1

for step in range(15):
    loss = (prediction - target) ** 2

    prediction.grad = 0.0
    loss.backward()

    prediction.data -= learning_rate * prediction.grad

    print(f"step {step}: prediction={prediction.data:.4f}, loss={loss.data:.4f}, grad={prediction.grad:.4f}")