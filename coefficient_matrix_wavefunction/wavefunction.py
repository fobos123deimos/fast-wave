# BSD 3-Clause License
#
# Copyright (c) 2024, Pikachu
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
    ```
    >>> hermite_sympy(2)
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
    ```
    >>> create_hermite_coefficients_table(3)
    array([[  0.,   0.,   0.,   1.],
          [  0.,   0.,   2.,   0.],
          [  0.,   4.,   0.,  -2.],
          [  8.,   0., -12.,   0.]])
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
def wavefunction_scipy(n: np.uint64, x: Union[np.float64, np.ndarray[np.float64]]) -> Union[np.float64, np.ndarray[np.float64]]:
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
    >>> wavefunction_scipy(1, 2.0)
    0.28752033217907963
    >>> wavefunction_scipy(1, np.array([0.0, 1.0, 2.0]))
    array([0.        , 0.64428837, 0.28752033])
    ```

    References
    ----------
    - Scipy Documentation on `eval_hermite`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html
    """

    return ((2 ** (-0.5 * n)) * (factorial(n) ** (-0.5)) * (np.pi ** (-0.25))) * np.exp(-(x ** 2) / 2) * eval_hermite(n, x)

@nb.jit(nopython=True, nogil=True, boundscheck=False, cache=True)
def wavefunction_c_matrix_value(n: np.uint64, x: Union[np.float64,np.complex128]) -> Union[np.float64,np.complex128]:
    """
    Compute the wavefunction using a precomputed matrix of Hermite polynomial coefficients.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : Union[np.float64,np.complex128]
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    Union[np.float64,np.complex128]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> wavefunction_c_matrix_value(0, 1.0)
    0.45558067201133257
    >>> wavefunction_c_matrix_value(0, 1.0 + 2.0j)
    (-1.4008797330262455-3.0609780602975003j)
    ```

    References
    ----------
    - Griffiths, D. J. (2005). Introduction to Quantum Mechanics (2nd Ed.). Pearson Education.
    """
    

    c_size = c_matrix.shape[0]
    coeffs = c_matrix[n]
    x_power = np.power(x,np.array([[c_size - i - 1 for i in range(c_size)]], dtype=np.float64).T)
    return (np.sum(x_power * coeffs[np.newaxis, :].T, axis=0) * np.exp(
        -(x**2) / 2) * math.pow(2, -0.5 * n) * math.pow(np.pi, -0.25) * math.pow(
            math.gamma(n + 1), -0.5))[0]

@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def wavefunction_c_matrix_vector(n: np.uint64, x: Union[np.ndarray[np.float64],np.ndarray[np.complex128]],
x_values: Union[np.ndarray[np.float64],np.ndarray[np.complex128]]) -> Union[np.ndarray[np.float64],np.ndarray[np.complex128]]:
    """
    Compute the wavefunction using a precomputed matrix of Hermite polynomial coefficients.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : Union[np.ndarray[np.float64],np.ndarray[np.complex128]]
        Position(s) at which to evaluate the wavefunction.
    x_values : Union[np.ndarray[np.float64],np.ndarray[np.complex128]]
        Preallocated array for results.

    Returns
    -------
    Union[np.ndarray[np.float64],np.ndarray[np.complex128]]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> wavefunction_c_matrix_vector(0, np.array([1.0,2.0]),np.zeros((1,x.shape[0]))[0])
    array([0.45558067, 0.10165379])
    >>> wavefunction_c_matrix_vector(0, np.array([1.0+1.0j,2.0+3.0j]),np.zeros((1,x.shape[0]),dtype=np.complex128)[0])
    array([0.40583486-0.63205035j, 8.78611733+2.55681454j])
    ```

    References
    ----------
    - Griffiths, D. J. (2005). Introduction to Quantum Mechanics (2nd Ed.). Pearson Education.
    """

    c_size = c_matrix.shape[0]
    coeffs = c_matrix[n]
    x_size = x.shape[0]

    for i in range(x_size):
        aux = 0.0
        for j in range(c_size):
            aux += coeffs[j] * pow(x[i], c_size - j - 1)

        x_values[i] = aux * np.exp(-(x[i]**2) / 2)

    return x_values * math.pow(2, -0.5 * n) * math.pow(math.gamma(n + 1),-0.5) * math.pow(np.pi, -0.25)

def wavefunction_real_value(n: np.uint64, x: np.float64) -> np.float64:
    """
    Wrapper function to compute the wavefunction to real values, choosing the appropriate method based on n.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.float64
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    np.float64
        The evaluated wavefunction real values, using either scipy or the coefficient matrix method.

    Examples
    --------
    ```python
    >>> wavefunction_real_value(0, 1.0)
    0.45558067201133257
    >>> wavefunction_real_value(61, 1.0)
    -0.23930491991711444
    ```
    """

    return  wavefunction_c_matrix_value(n,x) if(n<=60) else wavefunction_scipy(n,x)

def wavefunction_real_vector(n: np.uint64, x: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Wrapper function to compute the wavefunction to real vectors, choosing the appropriate method based on n.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.float64]
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    np.ndarray[np.float64]
        The evaluated wavefunction real vectors, using either scipy or the coefficient matrix method.

    Examples
    --------
    ```python
    >>> wavefunction_real_vector(0, np.array([1.0,2.0]))
    array([0.45558067, 0.10165379])
    >>> wavefunction_real_vector(61, np.array([1.0,2.0]))
    array([-0.23930492, -0.01677378])
    ```
    """
    return  wavefunction_c_matrix_vector(n,x,np.zeros((1,x.shape[0]))[0]) if(n<=60) else wavefunction_scipy(n,x)

def wavefunction_complex_value(n: np.uint64, x: np.complex128) -> np.complex128:
    """
    Wrapper function to compute the wavefunction to complex values, choosing the appropriate method based on n.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.complex128
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    np.complex128
        The evaluated wavefunction complex values, using either scipy or the coefficient matrix method.

    Examples
    --------
    ```python
    >>> wavefunction_complex_value(0, 1.0+1.0j)
    (0.4058348636708703-0.6320503516152827j)
    ```
    """
    if(n<=60):
        return  wavefunction_c_matrix_value(n,x)
    else:
        raise ValueError("This function is not enabled for complex x values ​​where n > 60.")
    
def wavefunction_complex_vector(n: np.uint64, x: np.complex128) -> np.complex128:
    """
    Wrapper function to compute the wavefunction to complex vectors, choosing the appropriate method based on n.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.complex128
        Position(s) at which to evaluate the wavefunction.

    Returns
    -------
    np.complex128
        The evaluated wavefunction complex vectors, using either scipy or the coefficient matrix method.

    Examples
    --------
    ```python
    >>> wavefunction_complex_vector(0, np.array([1.0+1.0j,2.0+3.0j]))
    array([0.40583486-0.63205035j, 8.78611733+2.55681454j])
    ```
    """
    if(n<=60):
        return  wavefunction_c_matrix_vector(n,x,np.zeros((1,x.shape[0]),dtype=np.complex128)[0])
    else:
        raise ValueError("This function is not enabled for complex x values ​​where n > 60.")

        
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
    test_output_scipy_value = wavefunction_scipy(0,1.0)
    test_output_scipy_vector = wavefunction_scipy(0,np.array([0.0, 1.0, 2.0]))
    test_output_real_value_cm = wavefunction_c_matrix_value(0,1.0)
    test_output_complex_value_cm = wavefunction_c_matrix_value(0,1.0 + 2.0j)
    test_output_real_vector = wavefunction_c_matrix_vector(0, np.array([0.0, 1.0, 2.0]),np.array([0.0,0.0,0.0]))
    test_output_complex_vector = wavefunction_c_matrix_vector(0, np.array([0.0 + 0.0j, 1.0 + 1.0j]),np.array([0.0 + 0.0j,0.0 + 0.0j]))
    compilation_test = True
    print(f"Functionality Test Passed: {compilation_test}")
except Exception as e:
    compilation_test = False
    print(f"Numba Compilation Error: {e}")
