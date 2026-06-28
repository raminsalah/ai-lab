from value import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c
e = d.relu()
f = e ** 2

f.backward()

print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("e =", e)
print("f =", f)