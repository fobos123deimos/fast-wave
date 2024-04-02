import os
import pickle

import math as mt
from typing import Union

import numpy as np
import numba as nb
from scipy.special import factorial
from scipy.special import eval_hermite
from sympy import symbols, diff, exp, Poly, Add

nb.set_num_threads(nb.get_num_threads())

C_matrix = None
Compilation_test = None


def hermite_sympy(n: np.uint64) -> Add:

  x = symbols("x")
  if n == 0:
    return 1
  else:
    return ((-1)**n) * exp(x**2) * diff(exp(-x**2), x, n)


def create_hermite_coefficients_table(N: np.uint64) -> np.ndarray[np.float64]:

  x = symbols("x")
  C = np.zeros((N + 1, N + 1), dtype=np.float64)
  C[0, N] = 1

  for n in range(1, N + 1):
    c = Poly(hermite_sympy(n), x).all_coeffs()
    for index in range(n, -1, -1):
      C[n, (N + 1) - index - 1] = float(c[n - index])

  return C



@nb.jit(forceobj=True, looplift=True, boundscheck=False)
def wavefunction_scipy_1D(n: np.uint64, x: Union[np.float64, np.ndarray[np.float64]]
) -> Union[np.float64, np.ndarray[np.float64]]:

  return (2**(-0.5 * n)) * (factorial(n)**(-0.5)) * (np.pi**(-0.25)) * np.exp(
      -(x**2) / 2) * eval_hermite(n, x)
  


@nb.jit(nopython=True,looplift=True,nogil=True,boundscheck=False,cache=True)
def wavefunction_c_matrix_1D(n: np.uint64, x: Union[np.float64, np.ndarray[np.float64],np.complex128,np.ndarray[np.complex128]],
    x_values: Union[np.ndarray[np.float64],np.ndarray[np.complex128],None] = None) -> np.ndarray[np.float64]:

  c_size = C_matrix.shape[0]
  coeffs = C_matrix[n]

  if(x_values == None):
    x_power = np.power(x,np.array([[c_size - i - 1 for i in range(c_size)]], dtype=np.float64).T)
  
    return np.sum(x_power * coeffs[np.newaxis, :].T, axis=0) * np.exp(
        -(x**2) / 2) * mt.pow(2, -0.5 * n) * mt.pow(np.pi, -0.25) * mt.pow(
            mt.gamma(n + 1), -0.5)

  x_size = x.shape[0]

  for i in range(x_size):
    aux = 0.0
    for j in range(c_size):
      aux += coeffs[j] * pow(x[i], c_size - j - 1)

    x_values[i] = aux * np.exp(-(x[i]**2) / 2)

  return x_values * mt.pow(2, -0.5 * n) * mt.pow(mt.gamma(n + 1),-0.5) * mt.pow(np.pi, -0.25)

def wavefunction_nx(n,x):
    
    is_Array = isinstance(x,np.ndarray)

    if(n<=60 and not(is_Array)):
        return wavefunction_c_matrix_1D(n,x)[0]
    
    elif(n<=60 and x.shape[0]<=50):
      x_values = np.zeros((1,x.shape[0]),dtype=np.complex128)[0]  if(isinstance(x[0],np.complex128)) else np.zeros((1,x.shape[0]))[0] 
      return wavefunction_c_matrix_1D(n,x,x_values)

    else:
        if(not(is_Array) and isinstance(x,complex)):
          raise ValueError("This function is not enabled for complex x values ​​where n > 60.")
        
        elif(is_Array and isinstance(x[0],np.complex128)):
          raise ValueError("This function is not enabled for complex x values ​​where n > 60.")
        
        else:
          return wavefunction_scipy_1D(n,x)


if(os.path.isfile("./coefficient_matrix_wavefunction/C_matrix.pickle")):
  with open("./coefficient_matrix_wavefunction/C_matrix.pickle", 'rb') as f:
    C_matrix = pickle.load(f)
else:
  C_matrix = create_hermite_coefficients_table(60)
  with open("./coefficient_matrix_wavefunction/C_matrix.pickle", 'wb') as f:
        pickle.dump(C_matrix, f)

try:
  t0 = wavefunction_nx(0,0.0)
  t1 = wavefunction_nx(61,0.0)
  Compilation_test = True

except Exception:
    Compilation_test = False
    print("Numba Compilation Error.")
