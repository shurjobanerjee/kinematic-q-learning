import autograd.numpy as np 
import autograd 

def f(x):
    A = np.arange(9, dtype=np.float).reshape(3,3)
    return np.dot(A, x)

jac = autograd.jacobian(f)
x = np.arange(3, dtype=np.float)
print(f(x))
print(jac(x))

