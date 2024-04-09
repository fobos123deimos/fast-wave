import os
import pickle
import math
from typing import Union

import numpy as np
import numba as nb
from scipy.special import factorial, eval_hermite
from sympy import symbols, diff, exp, Poly

# Adjusting numba settings to optimize parallel computations
nb.set_num_threads(nb.get_num_threads())

# Global variables for coefficient matrix and compilation status check
c_matrix = None
compilation_test = None

def hermite_sympy(n: np.uint64) -> Poly:
    """
    Compute the nth Hermite polynomial using symbolic differentiation.

    Parameters
    ----------
    n : np.uint64
        Order of the Hermite polynomial.

    Returns
    -------
    Poly
        The nth Hermite polynomial as a sympy expression.

    Examples
    --------
    ```python
    # Input
    hermite_sympy(2)

    # Output
    4*x**2 - 2
    ```

    References
    ----------
    - Wikipedia contributors. (2021). Hermite polynomials. In Wikipedia, The Free Encyclopedia. Retrieved from https://en.wikipedia.org/wiki/Hermite_polynomials
    """
    x = symbols("x")
    return 1 if n == 0 else ((-1) ** n) * exp(x ** 2) * diff(exp(-x ** 2), x, n)


def create_hermite_coefficients_table(n_max: np.uint64) -> np.ndarray:
    """
    Create a table of coefficients for Hermite polynomials up to order `n_max`.

    Parameters
    ----------
    n_max : np.uint64
        The maximum order of Hermite polynomials to compute.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the coefficients for the Hermite polynomials.

    Examples
    --------
    ```python
    # Input
    create_hermite_coefficients_table(3)

    # Output
    array([[ 0.,  0.,  0.,  1.],
           [ 0.,  0., -2.,  0.],
           [ 0.,  4.,  0., -2.],
           [ 8.,  0., -12., 0.]])
    ```

    References
    ----------
    - Olver, F. W. J., & Maximon, L. C. (2010). NIST Handbook of Mathematical Functions. Cambridge University Press.
    - NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.0.28 of 2020-09-15.
    - Sympy Documentation: https://docs.sympy.org/latest/modules/polys/index.html
    """
    x = symbols("x")
    C = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    C[0, n_max] = 1

    for n in range(1, n_max + 1):
        c = Poly(hermite_sympy(n), x).all_coeffs()
        for index in range(n, -1, -1):
            C[n, (n_max + 1) - index - 1] = float(c[n - index])

    return C


@nb.jit(forceobj=True, looplift=True, boundscheck=False)
def wavefunction_scipy_1d(n: np.uint64, x: Union[np.float64, np.ndarray]) -> Union[np.float64, np.ndarray]:
    """
    Compute the wavefunction for a given quantum state and position(s) using scipy.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : Union[np.float64, np.ndarray]
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    Union[np.float64, np.ndarray]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> wavefunction_scipy_1d(1, 2.0)
    0.28752033217907963
    >>> wavefunction_scipy_1d(1, np.array([0.0, 1.0, 2.0]))
    array([0.        , 0.64428837, 0.28752033])
    ```

    References
    ----------
    - Scipy Documentation on `eval_hermite`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html
    """

    return ((2 ** (-0.5 * n)) * (factorial(n) ** (-0.5)) * (np.pi ** (-0.25))) * np.exp(-(x ** 2) / 2) * eval_hermite(n, x)

@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def wavefunction_c_matrix_1D(n: np.uint64, x: Union[np.float64, np.ndarray[np.float64],np.complex128,np.ndarray[np.complex128]],
    x_values: Union[np.ndarray[np.float64],np.ndarray[np.complex128],None] = None) -> np.ndarray[np.float64]:
    """
    Compute the wavefunction using a precomputed matrix of Hermite polynomial coefficients.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : Union[np.float64, np.ndarray]
        Position(s) at which to evaluate the wavefunction.
    x_values : Union[np.ndarray, None], optional
        Optional, preallocated array for results.

    Returns
    -------
    np.ndarray
        The evaluated wavefunction.

    Examples
    --------
    ```
    >>> wavefunction_c_matrix_1D(0, 1.0)
    array([0.45558067])
    >>> wavefunction_c_matrix_1D(0, 1.0 + 2.0j)
    array([-1.40087973-3.06097806j])
    >>> wavefunction_c_matrix_1D(0, np.array([0.0, 1.0, 2.0]),np.array([0.0,0.0,0.0]))
    array([0.75112554, 0.45558067, 0.10165379])
    >>> wavefunction_c_matrix_1D(0, np.array([0.0 + 0.0j, 1.0 + 1.0j]),np.array([0.0 + 0.0j,0.0 + 0.0j]))
    array([0.75112554+0.j        , 0.40583486-0.63205035j])
    ```

    References
    ----------
    - Griffiths, D. J. (2005). Introduction to Quantum Mechanics (2nd Ed.). Pearson Education.
    """
    

    c_size = c_matrix.shape[0]
    coeffs = c_matrix[n]

    if(x_values is None):
        x_power = np.power(x,np.array([[c_size - i - 1 for i in range(c_size)]], dtype=np.float64).T)
        return np.sum(x_power * coeffs[np.newaxis, :].T, axis=0) * np.exp(
            -(x**2) / 2) * math.pow(2, -0.5 * n) * math.pow(np.pi, -0.25) * math.pow(
                math.gamma(n + 1), -0.5)

    x_size = x.shape[0]

    for i in range(x_size):
        aux = 0.0
        for j in range(c_size):
            aux += coeffs[j] * pow(x[i], c_size - j - 1)

        x_values[i] = aux * np.exp(-(x[i]**2) / 2)

    return x_values * math.pow(2, -0.5 * n) * math.pow(math.gamma(n + 1),-0.5) * math.pow(np.pi, -0.25)


def wavefunction_nx(n, x):
    """
    Wrapper function to compute the wavefunction, choosing the appropriate method based on the parameters.

    Parameters
    ----------
    n : int
        Quantum state number.
    x : Union[np.float64, np.ndarray]
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    The evaluated wavefunction, using either scipy or the coefficient matrix method.

    Examples
    --------
    ```
    >>> wavefunction_nx(0, 1.0)
    0.45558067201133257
    >>> wavefunction_nx(0, 1.0+1.0j)
    (0.4058348636708703-0.6320503516152827j)
    >>> wavefunction_nx(0, np.array([1.0,2.0]))
    array([0.45558067, 0.10165379])
    >>> wavefunction_nx(0, np.array([1.0+1.0j,2.0+3.0j]))
    array([0.40583486-0.63205035j, 8.78611733+2.55681454j])
    >>> wavefunction_nx(61, 2.0)
    -0.01677378220489314
    >>> wavefunction_nx(61, np.array([1.0,2.0]))
    array([-0.23930492, -0.01677378])
    ```
    """
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
          return wavefunction_scipy_1d(n,x)
        

"""
Main execution block to initialize the coefficient matrix and test the wavefunction computation.
This block checks for the existence of the precomputed Hermite polynomial coefficients matrix. If it doesn't exist,
it computes the matrix and saves it for future use. Then, it performs a basic test to verify that the wavefunction
computation works as expected.
References
----------
- Python Documentation: https://docs.python.org/3/tutorial/modules.html
    Python's official documentation offers insights into best practices for structuring and executing Python scripts.
"""
matrix_path = "./coefficient_matrix_wavefunction/C_matrix.pickle"
if os.path.isfile(matrix_path):
    with open(matrix_path, 'rb') as file:
        c_matrix = pickle.load(file)
else:
    c_matrix = create_hermite_coefficients_table(60)
    with open(matrix_path, 'wb') as file:
        pickle.dump(c_matrix, file)


try:

    # Basic functionality test
    test_output_0 = wavefunction_nx(0, 0.0)
    test_output_61 = wavefunction_nx(61, 0.0)
    compilation_test = True
    print(f"Functionality Test Passed: {compilation_test}")
except Exception as e:
    compilation_test = False
    print(f"Numba Compilation Error: {e}")
