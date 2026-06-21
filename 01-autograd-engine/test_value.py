from value import Value

x1 = Value(-2.0)
y1 = x1.relu()

print("Negative input:")
print("x1 =", x1)
print("y1 =", y1)

y1.backward()

print("After backward:")
print("x1 =", x1)
print()

x2 = Value(3.0)
y2 = x2.relu()

print("Positive input:")
print("x2 =", x2)
print("y2 =", y2)

y2.backward()

print("After backward:")
print("x2 =", x2)