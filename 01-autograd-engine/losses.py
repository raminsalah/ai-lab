from value import Value


def mse_loss(predictions, targets):
    losses = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    return sum(losses, Value(0.0)) / Value(len(losses))