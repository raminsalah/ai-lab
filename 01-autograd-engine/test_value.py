from value import Value

a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a

print("forward:")
print("a =", a)
print("b =", b)
print("c = a * b =", c)
print("d = c + a =", d)

d.backward()

print("\nafter backward:")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)