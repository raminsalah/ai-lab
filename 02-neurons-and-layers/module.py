class Module:
    """Base class for trainable neural-network components."""

    def __init__(self):
        self.training = True

    def parameters(self):
        """Return all trainable parameters owned by this module."""
        return []

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for parameter in self.parameters():
            parameter.grad = 0.0

    def train(self):
        """Switch this module to training mode."""
        self.training = True
        return self

    def eval(self):
        """Switch this module to evaluation mode."""
        self.training = False
        return self