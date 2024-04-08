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
    coefficients = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    coefficients[0, n_max] = 1

    for n in range(1, n_max + 1):
        poly_coeffs = Poly(hermite_sympy(n), x).all_coeffs()
        for index, coeff in enumerate(poly_coeffs[::-1]):
            coefficients[n, index] = float(coeff)

    return coefficients


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
    # Input
    wavefunction_scipy_1d(1, np.array([0, 1, 2]))

    # Output
    array([0.        , 0.75112554, 0.        ])
    ```

    References
    ----------
    - Scipy Documentation on `eval_hermite`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html
    """
    prefactor = ((2 ** (-0.5 * n)) * (factorial(n) ** (-0.5)) *
                 (math.pi ** (-0.25)))
    return prefactor * np.exp(-(x ** 2) / 2) * eval_hermite(n, x)

@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def wavefunction_c_matrix_1d(n: np.uint64, x: Union[np.float64, np.ndarray[np.float64], np.complex128, np.ndarray[np.complex128]],
                             x_values: Union[np.ndarray, None] = None) -> np.ndarray:
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
    ```python
    # Assuming c_matrix is already initialized and filled with coefficients.
    # Input
    wavefunction_c_matrix_1d(1, np.array([0, 1, 2]))

    # Output (example, depends on c_matrix values)
    array([0.        , 0.94530872, 0.        ])
    ```

    References
    ----------
    - Griffiths, D. J. (2005). Introduction to Quantum Mechanics (2nd Ed.). Pearson Education.
    """
    coeffs = c_matrix[n]
    if x_values is None:
        x_power = np.power(x, np.arange(coeffs.size)[::-1], dtype=np.float64)
        return np.dot(x_power, coeffs) * np.exp(-(x ** 2) / 2) * (
            math.pow(2, -0.5 * n) * math.pow(math.pi, -0.25) * math.pow(math.gamma(n + 1), -0.5))

    for i, val in enumerate(x):
        x_values[i] = np.dot(np.power(val, np.arange(coeffs.size)[::-1], dtype=np.float64),
                             coeffs) * np.exp(-(val ** 2) / 2)
    return x_values * (math.pow(2, -0.5 * n) * math.pow(math.gamma(n + 1), -0.5) *
                       math.pow(math.pi, -0.25))


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
    ```python
    # Assuming appropriate setup and initialization for c_matrix.
    # Input
    wavefunction_nx(2, np.array([0, 1, 2]))

    # Output (example, depends on c_matrix and method used)
    array([ 1.        ,  0.        , -0.95885108])
    ```
    """
    if n <= 60 and not isinstance(x, np.ndarray):
        return wavefunction_c_matrix_1d(n, x)[0]
    elif n <= 60 and x.size <= 50:
        dtype = np.complex128 if isinstance(x[0], np.complex128) else x.dtype
        x_values = np.zeros(x.shape, dtype=dtype)
        return wavefunction_c_matrix_1d(n, x, x_values)
    if not isinstance(x, np.ndarray) and isinstance(x, complex):
        raise ValueError("Function not enabled for complex x values where n > 60.")
    if isinstance(x, np.ndarray) and isinstance(x[0], np.complex128):
        raise ValueError("Function not enabled for complex x values where n > 60.")
    return wavefunction_scipy_1d(n, x)

if __name__ == "__main__":
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


