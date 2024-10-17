# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache, wraps
import numpy as np

def int_array_cache_Numba_single_fock(fn):

    """
    Cache decorator for functions that receive real multiple positions (numpy array) and is a problem Single Fock in the Numba module.

    This decorator caches function results to improve performance, particularly when `fn` is called 
    multiple times with the same arguments. The function to be decorated must accept an integer `n`, 
    a numpy array `x_array`, and a boolean `CS_matrix`. The numpy array is converted to a tuple for 
    caching purposes, as `lru_cache` only accepts hashable types.

    Parameters
    ----------
    fn : callable
        The function to be decorated, which takes three arguments:
        - n: np.uint64 representing the state number.
        - x_array: np.ndarray with dtype=np.float64 representing the positions.
        - CS_matrix: bool.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.

    .. note::

        This code is a modified version of the tensor_int_cache provided in Mr Mustard <https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26>`_,
        which is released under Apache License, Version 2.0 , with the following
        copyright notice:

        Copyright 2022 Xanadu Quantum Technologies Inc. All rights reserved.
    
    """

    @lru_cache
    def cached_wrapper(n, x_tuple,CS_matrix = True):
        x_array = np.array(x_tuple, dtype=np.float64)
        return fn(n,x_array,CS_matrix)

    @wraps(fn)
    def wrapper(n, x_array,CS_matrix = True):
        return cached_wrapper(n, tuple(x_array),CS_matrix)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def int_array_cache_Numba_multiple_fock(fn):

    """
    Cache decorator for functions that receive real multiple positions (numpy array) and is a problem Multiple Fock in the Numba module.

    This decorator caches function results to improve performance, particularly when `fn` is called 
    multiple times with the same arguments. The function to be decorated must accept an integer `n`, 
    a numpy array `x_array`, and a boolean `CS_matrix`. The numpy array is converted to a tuple for 
    caching purposes, as `lru_cache` only accepts hashable types.

    Parameters
    ----------
    fn : callable
        The function to be decorated, which takes three arguments:
        - n: np.uint64 representing the state number.
        - x_array: np.ndarray with dtype=np.float64 representing the positions.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.

    .. note::

        This code is a modified version of the tensor_int_cache provided in Mr Mustard <https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26>`_,
        which is released under Apache License, Version 2.0 , with the following
        copyright notice:

        Copyright 2022 Xanadu Quantum Technologies Inc. All rights reserved.
    
    """

    @lru_cache
    def cached_wrapper(n, x_tuple):
        x_array = np.array(x_tuple, dtype=np.float64)
        return fn(n,x_array)

    @wraps(fn)
    def wrapper(n, x_array):
        return cached_wrapper(n, tuple(x_array))

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def int_array_cache_Cython(fn):

    """
    Cache decorator for functions that receive real multiple positions (numpy array) in the Cython module.

    This decorator caches function results to improve performance, particularly when `fn` is called 
    multiple times with the same arguments. The function to be decorated must accept an integer `n` and
    a numpy array `x_array`. The numpy array is converted to a tuple for 
    caching purposes, as `lru_cache` only accepts hashable types.

    Parameters
    ----------
    fn : callable
        The function to be decorated, which takes three arguments:
        - n: np.uint64 representing the state number.
        - x_array: np.ndarray with dtype=np.float64 representing the positions.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.

    .. note::

        This code is a modified version of the tensor_int_cache provided in Mr Mustard <https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26>`_,
        which is released under Apache License, Version 2.0 , with the following
        copyright notice:

        Copyright 2022 Xanadu Quantum Technologies Inc. All rights reserved.
    
    """

    @lru_cache
    def cached_wrapper(n, x_tuple):
        x_array = np.array(x_tuple, dtype=np.float64)
        return fn(n,x_array)

    @wraps(fn)
    def wrapper(n, x_array):
        return cached_wrapper(n, tuple(x_array))

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def int_array_cache_Numba_complex_single_fock(fn):

    """
    Cache decorator for functions that receive complex multiple positions (numpy array) and is a problem Single Fock in the Numba module.

    This decorator caches function results to improve performance, particularly when `fn` is called 
    multiple times with the same arguments. The function to be decorated must accept an integer `n`, 
    a numpy array `x_array`, and a boolean `CS_matrix`. The numpy array is converted to a tuple for 
    caching purposes, as `lru_cache` only accepts hashable types.

    Parameters
    ----------
    fn : callable
        The function to be decorated, which takes three arguments:
        - n: np.uint64 representing the state number.
        - x_array: np.ndarray with dtype=np.complex128 representing the positions.
        - CS_matrix: bool.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.

    .. note::

        This code is a modified version of the tensor_int_cache provided in Mr Mustard <https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26>`_,
        which is released under Apache License, Version 2.0 , with the following
        copyright notice:

        Copyright 2022 Xanadu Quantum Technologies Inc. All rights reserved.

    """

    @lru_cache
    def cached_wrapper(n, x_tuple,CS_matrix):
        x_array = np.array(x_tuple, dtype=np.complex128)
        return fn(n,x_array,CS_matrix)

    @wraps(fn)
    def wrapper(n, x_array,CS_matrix):
        return cached_wrapper(n, tuple(x_array),CS_matrix)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def int_array_cache_Numba_complex_multiple_fock(fn):

    """
    Cache decorator for functions that receive complex multiple positions (numpy array)  in the Numba module.

    This decorator caches function results to improve performance, particularly when `fn` is called 
    multiple times with the same arguments. The function to be decorated must accept an integer `n`, 
    a numpy array `x_array`, and a boolean `CS_matrix`. The numpy array is converted to a tuple for 
    caching purposes, as `lru_cache` only accepts hashable types.

    Parameters
    ----------
    fn : callable
        The function to be decorated, which takes three arguments:
        - n: np.uint64 representing the state number.
        - x_array: np.ndarray with dtype=np.complex128 representing the positions.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.

    .. note::

        This code is a modified version of the tensor_int_cache provided in Mr Mustard <https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26>`_,
        which is released under Apache License, Version 2.0 , with the following
        copyright notice:

        Copyright 2022 Xanadu Quantum Technologies Inc. All rights reserved.

    """

    @lru_cache
    def cached_wrapper(n, x_tuple):
        x_array = np.array(x_tuple, dtype=np.complex128)
        return fn(n,x_array)

    @wraps(fn)
    def wrapper(n, x_array):
        return cached_wrapper(n, tuple(x_array))

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear


def int_array_cache_Cython_complex(fn):

    """
    Cache decorator for functions that receive real multiple positions (numpy array) and is a problem Single Fock in the Cython module.

    This decorator caches function results to improve performance, particularly when `fn` is called 
    multiple times with the same arguments. The function to be decorated must accept an integer `n` and
    a numpy array `x_array`. The numpy array is converted to a tuple for 
    caching purposes, as `lru_cache` only accepts hashable types.

    Parameters
    ----------
    fn : callable
        The function to be decorated, which takes three arguments:
        - n: np.uint64 representing the state number.
        - x_array: np.ndarray with dtype=np.complex128 representing the positions.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.

    .. note::

        This code is a modified version of the tensor_int_cache provided in Mr Mustard <https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26>`_,
        which is released under Apache License, Version 2.0 , with the following
        copyright notice:

        Copyright 2022 Xanadu Quantum Technologies Inc. All rights reserved.

    """

    @lru_cache
    def cached_wrapper(n, x_tuple):
        x_array = np.array(x_tuple, dtype=np.complex128)
        return fn(n,x_array)

    @wraps(fn)
    def wrapper(n, x_array):
        return cached_wrapper(n, tuple(x_array))

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
