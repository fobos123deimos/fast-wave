from functools import lru_cache, wraps
import numpy as np

def int_array_cache_Numba(fn):

    """
    Cache decorator for functions that receive real multiple positions (numpy array) in the Numba module.

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
    """

    @lru_cache
    def cached_wrapper(n, x_tuple,CS_matrix):
        x_array = np.array(x_tuple, dtype=np.float64)
        return fn(n,x_array,CS_matrix)

    @wraps(fn)
    def wrapper(n, x_array,CS_matrix):
        return cached_wrapper(n, tuple(x_array),CS_matrix)

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


def int_array_cache_Numba_complex(fn):

    """
    Cache decorator for functions that receive complex multiple positions (numpy array) in the Numba module.

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

def int_array_cache_Cython_complex(fn):

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
        - x_array: np.ndarray with dtype=np.complex128 representing the positions.

    Returns
    -------
    callable
        A wrapped version of `fn` with caching enabled, including methods to access cache information:
        - cache_info: Returns cache statistics.
        - cache_clear: Clears the cache.
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
