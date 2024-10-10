# BSD 3-Clause License
#
# Copyright (c) 2024, Matheus Gomes Cordeiro 
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

import math
from sympy import symbols, diff, exp, Poly

import numpy as np
import numba as nb

# Adjusting numba settings to optimize parallel computations
nb.set_num_threads(nb.get_num_threads())

# Global variables for coefficient matrix and compilation status check
c_s_matrix = None
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


def create_normalized_hermite_coefficients_matrix(n_max: np.uint64) -> np.ndarray:
    """
    Create a matrix of coefficients for  normalized Hermite polynomials up to order `n_max`.

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
    >>> create_normalized_hermite_coefficients_matrix(3)
    array([[ 0.        ,  0.        ,  0.        ,  0.75112554],
           [ 0.        ,  0.        ,  1.06225193,  0.        ],
           [ 0.        ,  1.06225193,  0.        , -0.53112597],
           [ 0.86732507,  0.        , -1.30098761,  0.        ]])
    ```

    References
    ----------
    - Olver, F. W. J., & Maximon, L. C. (2010). NIST Handbook of Mathematical Functions. Cambridge University Press.
    - NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.0.28 of 2020-09-15.
    - Sympy Documentation: https://docs.sympy.org/latest/modules/polys/index.html
    """
    x = symbols("x")
    C_s = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    C_s[0, n_max] = 1

    for n in range(1, n_max + 1):
        c = Poly(hermite_sympy(n), x).all_coeffs()
        for index in range(n, -1, -1):
            C_s[n, (n_max + 1) - index - 1] = float(c[n - index])

    for i in range(n_max + 1):
        C_s[i] /=  (np.pi**0.50 * (2**i) * math.gamma(i+1))**0.5

    return C_s 


@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def psi_n_single_fock_single_position(n: np.uint64, x:np.float64, CS_matrix:bool = True) -> np.float64:

    """
    Compute the wavefunction to a real scalar x using a pre-computed matrix of normalized Hermite polynomial coefficients until n=60 and 
    then use the adapted recurrence relation for higher orders.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.float64
        Position(s) at which to evaluate the wavefunction.
    more_fast : bool, optional
        If True, use the optimized method for n <= 60, which relies on a pre-computed matrix of coefficients for faster computation. 
        For n > 60 or if False, use the general recursion method. Defaults to True.


    Returns
    -------
        np.float64
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_single_position(0, 1.0)
    0.45558067201133257
    >>> psi_n_single_fock_single_position(61, 1.0)
    -0.2393049199171131
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    if(n<=60 and CS_matrix):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = 0.0
        for i in range(c_size-n-1,c_size,2):
            result += n_coeffs[i]*(x**(c_size-i-1))

        return result * np.exp(-(x ** 2) / 2)
    else:
        r0 = 0.0
        r1 = (np.pi ** (-0.25)) * np.exp(-(x ** 2) / 2)
        r2 = 0.0

        for index in range(n):
            r2 = 2 * x * (r1 / np.sqrt(2 * (index + 1))) - np.sqrt(index / (index + 1)) * r0 
            r0 = r1
            r1 = r2

        return r1
    

@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def psi_n_single_fock_single_position_complex(n: np.uint64, x: np.complex128, CS_matrix:bool = True) -> np.complex128:

    """
    Compute the wavefunction to a complex scalar x using a pre-computed matrix of normalized Hermite polynomial coefficients until n=60 and 
    then use the adapted recurrence relation for higher orders.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.complex128
        Position(s) at which to evaluate the wavefunction.
    more_fast : bool, optional
        If True, use the optimized method for n <= 60, which relies on a pre-computed matrix of coefficients for faster computation. 
        For n > 60 or if False, use the general recursion method. Defaults to True.


    Returns
    -------
        np.complex128
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_single_position_complex(0,1.0+2.0j)
    (-1.4008797330262455-3.0609780602975003j)
    >>> psi_n_single_fock_single_position_complex(61,1.0+2.0j)
    (-511062135.47555304+131445997.75753704j)
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """

    if(n<=60 and CS_matrix):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = 0.0 + 0.0j
        for i in range(c_size-n-1,c_size,2):
            result += n_coeffs[i]*(x**(c_size-i-1))

        return result * np.exp(-(x ** 2) / 2)
    else:
        r0 = 0.0 + 0.0j
        r1 = (np.pi ** (-0.25)) * np.exp(-(x ** 2) / 2)
        r2 = 0.0 + 0.0j

        for index in range(n):
            r2 = 2 * x * (r1 / np.sqrt(2 * (index + 1))) - np.sqrt(index / (index + 1)) * r0 
            r0 = r1
            r1 = r2

        return r1


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def psi_n_single_fock_multiple_position(n: np.uint64, x: np.ndarray[np.float64], CS_matrix: bool = True) -> np.ndarray[np.float64]:

    """
    Compute the wavefunction to a real vector x using a pre-computed matrix of normalized Hermite polynomial coefficients until n=60 and 
    then use the adapted recurrence relation for higher orders.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.float64]
        Position(s) at which to evaluate the wavefunction.
    more_fast : bool, optional
        If True, use the optimized method for n <= 60, which relies on a pre-computed matrix of coefficients for faster computation. 
        For n > 60 or if False, use the general recursion method. Defaults to True.

   

    Returns
    -------
        np.ndarray[np.float64]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_multiple_position(0,np.array([1.0, 2.0]))
    array([0.45558067, 0.10165379])
    >>> psi_n_single_fock_multiple_position(61,np.array([1.0, 2.0]))
    array([-0.23930492, -0.01677378])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """

    x_size = x.shape[0]

    if(n<=60 and x_size<= 35 and CS_matrix):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = np.array([0.0] * (x_size))
        for j in range(x_size):
            for i in range(c_size-n-1,c_size,2):
                result[j] += n_coeffs[i]*(x[j]**(c_size-i-1))
            result[j] *= np.exp(-(x[j] ** 2) / 2)
        return result
    else:
        result = np.array([[0.0]*(x_size)]*(n+1))
        result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

        for index in range(n):
            result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
            
        return result[-1]
    


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def psi_n_single_fock_multiple_position_complex(n: np.uint64, x: np.ndarray[np.complex128], CS_matrix: bool = True) -> np.ndarray[np.complex128]:

    """
    Compute the wavefunction to a complex vector x using a pre-computed matrix of normalized Hermite polynomial coefficients until n=60 and 
    then use the adapted recurrence relation for higher orders.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.complex128]
        Position(s) at which to evaluate the wavefunction.
    more_fast : bool, optional
        If True, use the optimized method for n <= 60, which relies on a pre-computed matrix of coefficients for faster computation. 
        For n > 60 or if False, use the general recursion method. Defaults to True.

   

    Returns
    -------
        np.ndarray[np.complex128]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_multiple_position_complex(0,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([ 0.40583486-0.63205035j, -0.49096842+0.56845369j])
    >>> psi_n_single_fock_multiple_position_complex(61,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([-7.56548941e+03+9.21498621e+02j, -1.64189542e+08-3.70892077e+08j])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """

    x_size = x.shape[0]

    if(n<=60 and x_size<= 35 and CS_matrix):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = np.array([0.0 + 0.0j] * (x_size))
        for j in range(x_size):
            for i in range(c_size-n-1,c_size,2):
                result[j] += n_coeffs[i]*(x[j]**(c_size-i-1))
            result[j] *= np.exp(-(x[j] ** 2) / 2)
        return result
    else:
        result = np.array([[0.0 + 0.0j]*(x_size)]*(n+1))
        result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

        for index in range(n):
            result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
            
        return result[-1]


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def psi_n_multiple_fock_single_position(n: np.uint64, x:np.float64) -> np.ndarray[np.float64]:

    """
    Compute the wavefunction to a real scalar x to all fock states until n using the recurrence relation.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.float64
        Position(s) at which to evaluate the wavefunction.
   

    Returns
    -------
        np.ndarray[np.float64]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_single_position(1,1.0)
    array([0.45558067, 0.64428837])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    result = np.array([0.0] * (n+1))
    result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

    for index in range(n):
        result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
        
    return result


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def psi_n_multiple_fock_single_position_complex(n: np.uint64, x: np.complex128) -> np.ndarray[np.complex128]: 

    """
    Compute the wavefunction to a complex scalar x to all fock states until n using the recurrence relation.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.complex128
        Position(s) at which to evaluate the wavefunction.
   

    Returns
    -------
        np.ndarray[np.complex128]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_single_position_complex(1,1.0 +2.0j)
    array([-1.40087973-3.06097806j,  6.67661026-8.29116292j])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    result = np.array([0.0 + 0.0j] * (n+1))
    result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

    for index in range(n):
        result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
        
    return result



@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def psi_n_multiple_fock_multiple_position(n: np.uint64, x: np.ndarray[np.float64]) -> np.ndarray[np.ndarray[np.float64]]:

    """
    Compute the wavefunction to a real vector x to all fock states until n using the recurrence relation.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.float64]
        Position(s) at which to evaluate the wavefunction.
   

    Returns
    -------
        np.ndarray[np.ndarray[np.float64]]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_multiple_position(1,np.array([1.0, 2.0]))
    array([[0.45558067, 0.10165379],
           [0.64428837, 0.28752033]])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    result = np.array([[0.0]*(x_size)]*(n+1))
    result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

    for index in range(n):
        result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
        
    return result


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def psi_n_multiple_fock_multiple_position_complex(n: np.uint64, x: np.ndarray[np.complex128]) -> np.ndarray[np.ndarray[np.float64]]:

    """
    Compute the wavefunction to a complex vector x to all fock states until n using the recurrence relation.

    Parameters
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.complex128]
        Position(s) at which to evaluate the wavefunction.
   

    Returns
    -------
        np.ndarray[np.ndarray[np.complex128]]
        The evaluated wavefunction.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_multiple_position_complex(1,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
           [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    x_size = x.shape[0]
    result = np.array([[0.0 + 0.0j]*(x_size)]*(n+1))
    result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

    for index in range(n):
        result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
        
    return result
        
"""
Main execution block to initialize the coefficient matrix and test the wavefunction computation.

References
----------
- Python's official documentation offers insights into best practices for structuring and executing Python scripts:
  https://docs.python.org/3/tutorial/modules.html 
  .
"""

c_s_matrix = create_normalized_hermite_coefficients_matrix(60)

try:

    # Basic functionality test
    test_output_sfsp_CS_matrix_2 = psi_n_single_fock_single_position(2, 10.0)
    test_output_sfsp_2 = psi_n_single_fock_single_position(2, 10.0, CS_matrix=False)
    test_output_sfsp_CS_matrix_61 = psi_n_single_fock_single_position(61, 10.0)
    test_output_sfsp_61 = psi_n_single_fock_single_position(61, 10.0, CS_matrix=False)
    test_output_mfsp = psi_n_multiple_fock_single_position(2, 10.0)
    test_output_sfmp_CS_matrix_2 = psi_n_single_fock_multiple_position(2, np.array([10.0,4.5]))
    test_output_sfmp_2 = psi_n_single_fock_multiple_position(2, np.array([10.0,4.5]), CS_matrix=False)
    test_output_sfmp_CS_matrix_61 = psi_n_single_fock_multiple_position(61, np.array([10.0,4.5]))
    test_output_sfmp_61 = psi_n_single_fock_multiple_position(61, np.array([10.0,4.5]), CS_matrix=False)
    test_output_mfmp = psi_n_multiple_fock_multiple_position(2, np.array([10.0,4.5]))
    test_output_sfsp_c_CS_matrix_2 = psi_n_single_fock_single_position_complex(2, 10.0 + 0.0j)
    test_output_sfsp_c_2 = psi_n_single_fock_single_position_complex(2, 10.0 + 0.0j, CS_matrix=False)
    test_output_sfsp_c_CS_matrix_61 = psi_n_single_fock_single_position_complex(61, 10.0 + 0.0j)
    test_output_sfsp_c_61 = psi_n_single_fock_single_position_complex(61, 10.0 + 0.0j, CS_matrix=False)
    test_output_mfsp_c = psi_n_multiple_fock_single_position_complex(2, 10.0 + 0.0j)
    test_output_sfmp_c_CS_matrix_2 = psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j,4.5 + 0.0j]))
    test_output_sfmp_c_2 = psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j,4.5 + 0.0j]), CS_matrix=False)
    test_output_sfmp_c_CS_matrix_61 = psi_n_single_fock_multiple_position_complex(61, np.array([10.0 + 0.0j,4.5 + 0.0j]))
    test_output_sfmp_c_61 = psi_n_single_fock_multiple_position_complex(61, np.array([10.0 + 0.0j,4.5 + 0.0j]), CS_matrix=False)
    test_output_mfmp_c = psi_n_multiple_fock_multiple_position_complex(2, np.array([10.0 + 0.0j,4.5 + 0.0j]))
    compilation_test = True
    print(f"Functionality Test Passed: {compilation_test}")
except Exception as e:
    compilation_test = False
    print(f"Numba Compilation Error: {e}")
