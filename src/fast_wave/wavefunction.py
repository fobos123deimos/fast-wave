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

import os
import pickle
import math
from functools import lru_cache

import numpy as np
import numba as nb

# Adjusting numba settings to optimize parallel computations
nb.set_num_threads(nb.get_num_threads())

# Global variables for coefficient matrix and compilation status check
c_s_matrix = None
compilation_test = None

@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def create_normalized_hermite_coefficients_matrix(n_max: np.uint64) -> np.ndarray:
    """
    Create a matrix of normalized coefficients for Hermite polynomials up to order `n_max`.

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
    C_s = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)

    for i in range(n_max+1):
        for j in range(n_max+1):
            if((j>=(n_max-i)) and ((n_max-i+j)%2 == 0)):
                C_s[i,j] = ( ((-1)**((j-n_max+i)/2)) * (2**(n_max-j-(i*0.5))) ) / ( math.gamma(((j-n_max+i)/2) + 1) * math.gamma(n_max-j + 1) )
            else:
                C_s[i,j] = 0.0
        C_s[i] *= math.gamma(i+1)**0.5

    return C_s/(np.pi**0.25) 




@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def wavefunction_smod(n: np.uint64, x:np.float64, more_fast:bool = True) -> np.float64:

    """
    Compute the wavefunction to a real scalar x using a pre-computed matrix of Hermite polynomial normalized coefficients until n=60 and 
    then use the adapted recursion relation for multidimensional M-mode wavefunction for higher orders.

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
    >>> wavefunction_smod(0, 1.0)
    0.45558067201133257
    >>> wavefunction_smod(61, 1.0)
    -0.2393049199171131
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """
    
    if(n<=60 and more_fast):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = 0.0
        for i in range(c_size-n-1,c_size,2):
            c = n_coeffs[i]
            result += c*(x**(c_size-i-1))

        return result * np.exp(-(x ** 2) / 2)
    else:
        result = np.array([0.0] * (n+1))
        result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

        for index in range(n):
            result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
            
        return result[-1]
    

@nb.jit(nopython=True, looplift=True, nogil=True, boundscheck=False, cache=True)
def c_wavefunction_smod(n: np.uint64, x: np.complex128, more_fast:bool = True) -> np.complex128:

    """
    Compute the wavefunction to a complex scalar x using a pre-computed matrix of Hermite polynomial normalized coefficients until n=60 and 
    then use the adapted recursion relation for multidimensional M-mode wavefunction for higher orders.

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
    >>> c_wavefunction_smod(0,1.0+2.0j)
    (-1.4008797330262455-3.0609780602975003j)
    >>> c_wavefunction_smod(61,1.0+2.0j)
    (-511062135.47555304+131445997.75753704j)
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """

    if(n<=60 and more_fast):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = 0.0 + 0.0j
        for i in range(c_size-n-1,c_size,2):
            c = n_coeffs[i]
            result += c*(x**(c_size-i-1))

        return result * np.exp(-(x ** 2) / 2)
    else:
        result = np.array([0.0 + 0.0j] * (n+1))
        result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

        for index in range(n):
            result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
            
        return result[-1]


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def wavefunction_smmd(n: np.uint64, x: np.ndarray[np.float64], more_fast: bool = True) -> np.ndarray[np.float64]:

    """
    Compute the wavefunction to a real vector x using a pre-computed matrix of Hermite polynomial normalized coefficients until n=60 and 
    then use the adapted recursion relation for multidimensional M-mode wavefunction for higher orders.

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
    >>> wavefunction_smmd(0,np.array([1.0, 2.0]))
    array([0.45558067, 0.10165379])
    >>> wavefunction_smmd(61,np.array([1.0, 2.0]))
    array([-0.23930492, -0.01677378])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """

    x_size = x.shape[0]

    if(n<=60 and more_fast):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = np.array([0.0] * (x_size))
        for j in range(x_size):
            for i in range(c_size-n-1,c_size,2):
                c = n_coeffs[i]
                result[j] += c*(x[j]**(c_size-i-1))
            result[j] *= np.exp(-(x[j] ** 2) / 2)
        return result
    else:
        result = np.array([[0.0]*(x_size)]*(n+1))
        result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

        for index in range(n):
            result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
            
        return result[-1]
    


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def c_wavefunction_smmd(n: np.uint64, x: np.ndarray[np.complex128], more_fast: bool = True) -> np.ndarray[np.complex128]:

    """
    Compute the wavefunction to a complex vector x using a pre-computed matrix of Hermite polynomial normalized coefficients until n=60 and 
    then use the adapted recursion relation for multidimensional M-mode wavefunction for higher orders.

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
    >>> c_wavefunction_smmd(0,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([ 0.40583486-0.63205035j, -0.49096842+0.56845369j])
    >>> c_wavefunction_smmd(61,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
    array([-7.56548941e+03+9.21498621e+02j, -1.64189542e+08-3.70892077e+08j])
    ```

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 
      015402. doi:10.1088/1361-6404/aa9584
    """

    x_size = x.shape[0]

    if(n<=60 and more_fast):
        c_size = c_s_matrix.shape[0]
        n_coeffs = c_s_matrix[n]
        result = np.array([0.0 + 0.0j] * (x_size))
        for j in range(x_size):
            for i in range(c_size-n-1,c_size,2):
                c = n_coeffs[i]
                result[j] += c*(x[j]**(c_size-i-1))
            result[j] *= np.exp(-(x[j] ** 2) / 2)
        return result
    else:
        result = np.array([[0.0 + 0.0j]*(x_size)]*(n+1))
        result[0] = (np.pi ** (-0.25))*np.exp(-(x ** 2) / 2) 

        for index in range(n):
            result[index+1]  = 2*x*(result[index]/np.sqrt(2*(index+1))) - np.sqrt(index/(index+1)) * result[index-1]
            
        return result[-1]


@nb.jit(nopython=True, looplift=True,nogil=True, boundscheck=False, cache=True)
def wavefunction_mmod(n: np.uint64, x:np.float64) -> np.ndarray[np.float64]:

    """
    Compute the wavefunction to a real scalar x to all modes until the mode n using the recursion relation for multidimensional M-mode wavefunction.

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
    >>> wavefunction_mmod(1,1.0)
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
def c_wavefunction_mmod(n: np.uint64, x: np.complex128) -> np.ndarray[np.complex128]: 

    """
    Compute the wavefunction to a complex scalar x to all modes until the mode n using the recursion relation for multidimensional M-mode wavefunction.

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
    >>> c_wavefunction_mmod(1,1.0 +2.0j)
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
def wavefunction_mmmd(n: np.uint64, x: np.ndarray[np.float64]) -> np.ndarray[np.ndarray[np.float64]]:

    """
    Compute the wavefunction to a real vector x to all modes until the mode n using the recursion relation for multidimensional M-mode wavefunction.

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
    >>> wavefunction_mmmd(1,np.array([1.0, 2.0]))
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
def c_wavefunction_mmmd(n: np.uint64, x: np.ndarray[np.complex128]) -> np.ndarray[np.ndarray[np.float64]]:

    """
    Compute the wavefunction to a complex vector x to all modes until the mode n using the recursion relation for multidimensional M-mode wavefunction.

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
    >>> c_wavefunction_mmmd(1,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
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

def wavefunction(s_mode: bool = True, o_dimensional: bool = True, complex_bool: bool = False, cache: bool = False, cache_size: np.uint64 = 128) -> nb.core.registry.CPUDispatcher:

    """
    Computes the wavefunction of a quantum harmonic oscillator .This function dispatches to different implementations of the wavefunction depending on the specified parameters.

    Parameters:
    ----------
    s_mode : bool 
             Whether to use the single-mode recursion (True) or the multi-mode recursion (False). Defaults to True.
    o_dimensional : bool 
                    Whether to evaluate the wavefunction for a single coordinate (True) or multiple coordinates (False). Defaults to True.
    complex_boll : bool 
                   Whether to return the complex wavefunction (True) or the real wavefunction (False). Defaults to False.
    cache : bool  
            Whether to cache the results of the wavefunction computation using least-recently-used caching. Defaults to False.
    cache_size : np.uint64 
                 The maximum size of the cache to use. Defaults to 128.

    Returns:
    -------
        nb.core.registry.CPUDispatcher: 
        A dispatcher object that resolves to the appropriate wavefunction implementation based on the input parameters.
   """
    
    if(s_mode):
        if(o_dimensional):
            if(not(complex_bool)):
                if(not(cache)):
                    return wavefunction_smod
                else:
                    return lru_cache(maxsize=cache_size)(wavefunction_smod)
            else:
                if(not(cache)):
                    return c_wavefunction_smod
                else:
                    return lru_cache(maxsize=cache_size)(c_wavefunction_smod)
        else:
            if(not(complex_bool)):
                if(not(cache)):
                    return wavefunction_smmd
                else:
                    return lru_cache(maxsize=cache_size)(wavefunction_smmd)
            else:
                if(not(cache)):
                    return c_wavefunction_smmd
                else:
                    return lru_cache(maxsize=cache_size)(c_wavefunction_smmd)
    else:
        if(o_dimensional):
            if(not(complex_bool)):
                if(not(cache)):
                    return wavefunction_mmod
                else:
                    return lru_cache(maxsize=cache_size)(wavefunction_mmod)
            else:
                if(not(cache)):
                    return c_wavefunction_mmod
                else:
                    return lru_cache(maxsize=cache_size)(c_wavefunction_mmod)
        else:
            if(not(complex_bool)):
                if(not(cache)):
                    return wavefunction_mmmd
                else:
                    return lru_cache(maxsize=cache_size)(wavefunction_mmmd)
            else:
                if(not(cache)):
                    return c_wavefunction_mmmd
                else:
                    return lru_cache(maxsize=cache_size)(c_wavefunction_mmmd)
        
"""
Main execution block to initialize the coefficient matrix and test the wavefunction computation.
This block checks for the existence of the precomputed Hermite polynomial coefficients matrix. If it doesn't exist,
it computes the matrix and saves it for future use. Then, it performs a basic test to verify that the wavefunction
computation works as expected.

References
----------
- Python's official documentation offers insights into best practices for structuring and executing Python scripts:
  https://docs.python.org/3/tutorial/modules.html 
  .
"""
package_dir = os.path.dirname(__file__)  
matrix_filename = 'C_s_matrix.pickle'
matrix_path = os.path.join(package_dir, matrix_filename)

if os.path.isfile(matrix_path):
    with open(matrix_path, 'rb') as file:
        c_s_matrix = pickle.load(file)
else:
    c_s_matrix = create_normalized_hermite_coefficients_matrix(60)
    with open(matrix_path, 'wb') as file:
        pickle.dump(c_s_matrix, file)


try:

    # Basic functionality test
    test_output_smod = wavefunction_smod(2, 10.0)
    test_output_smod_less_fast = wavefunction_smod(2, 10.0, more_fast=False)
    test_output_mmod = wavefunction_mmod(2, 10.0)
    test_output_smmd = wavefunction_smmd(2, np.array([10.0,4.5]))
    test_output_smmd_less_fast = wavefunction_smmd(2, np.array([10.0,4.5]), more_fast=False)
    test_output_mmmd = wavefunction_mmmd(2, np.array([10.0,4.5]))
    test_output_c_smod = c_wavefunction_smod(2, 10.0 + 0.0j)
    test_output_c_smod_less_fast = c_wavefunction_smod(2, 10.0 + 0.0j, more_fast=False)
    test_output_c_mmod = c_wavefunction_mmod(2, 10.0 + 0.0j)
    test_output_c_smmd = c_wavefunction_smmd(2, np.array([10.0 + 0.0j,4.5 + 0.0j]))
    test_output_c_smmd_less_fast = c_wavefunction_smmd(2, np.array([10.0 + 0.0j,4.5 + 0.0j]), more_fast=False)
    test_output_c_mmmd = c_wavefunction_mmmd(2, np.array([10.0 + 0.0j,4.5 + 0.0j]))
    compilation_test = True
    print(f"Functionality Test Passed: {compilation_test}")
except Exception as e:
    compilation_test = False
    print(f"Numba Compilation Error: {e}")
