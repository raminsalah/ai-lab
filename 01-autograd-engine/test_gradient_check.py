from value import Value

h = 0.0001

x = Value(6.0)
y = x ** 2
y.backward()

autograd_grad = x.grad

x1 = Value(6.0)
y1 = x1 ** 2

x2 = Value(6.0 + h)
y2 = x2 ** 2

numerical_grad = (y2.data - y1.data) / h

print("autograd grad  :", autograd_grad)
print("numerical grad :", numerical_grad)
print("difference     :", abs(autograd_grad - numerical_grad))