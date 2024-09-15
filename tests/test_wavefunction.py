import pytest
import numpy as np
from src.fast_wave.wavefunction import *

@pytest.fixture(scope="module", autouse=True)
def initialize_c_s_matrix():
    """
    Fixture to initialize the global variable c_s_matrix before running tests.
    """
    global c_s_matrix
    c_s_matrix = create_normalized_hermite_coefficients_matrix(60)

def test_hermite_sympy():
    """
    Tests the hermite_sympy function to verify the accuracy of Hermite polynomial computation.
    """
    x = symbols("x")
    h0 = hermite_sympy(0)
    h1 = hermite_sympy(1)
    h2 = hermite_sympy(2)

    assert h0 == 1
    assert h1 == 2 * x
    assert h2 == 4 * x**2 - 2

def test_create_hermite_coefficients_table():
    """
    Tests the create_normalized_hermite_coefficients_table function to verify if the normalized coefficient matrix is correct.
    """
    n_max = 2
    coeffs_table = create_normalized_hermite_coefficients_matrix(n_max)
    
    expected_table = np.zeros((3, 3))
    expected_table[0, 2] = 0.75112554 
    expected_table[1, 1] = 1.06225193  
    expected_table[2, 0] = 1.06225193 
    expected_table[2, 2] = -0.53112597

    assert np.allclose(coeffs_table, expected_table)

def test_wavefunction_computation():
    """
    Tests the basic functionality of all wavefunction functions.
    """

    wave_smod = wavefunction(s_mode = True, o_dimensional = True, complex_bool = False, cache = False, cache_size = 128)
    wave_smmd = wavefunction(s_mode = True, o_dimensional = False, complex_bool = False, cache = False, cache_size = 128)
    wave_mmod = wavefunction(s_mode = False, o_dimensional = True, complex_bool = False, cache = False, cache_size = 128)
    wave_mmmd = wavefunction(s_mode = False, o_dimensional = False, complex_bool = False, cache = False, cache_size = 128)
    c_wave_smod = wavefunction(s_mode = True, o_dimensional = True, complex_bool = True, cache = False, cache_size = 128)
    c_wave_smmd = wavefunction(s_mode = True, o_dimensional = False, complex_bool = True, cache = False, cache_size = 128)
    c_wave_mmod = wavefunction(s_mode = False, o_dimensional = True, complex_bool = True, cache = False, cache_size = 128)
    c_wave_mmmd = wavefunction(s_mode = False, o_dimensional = False, complex_bool = True, cache = False, cache_size = 128)

    # Testing basic functionality
    test_output_odsm_2 = wave_smod(2, 10.0)
    assert isinstance(test_output_odsm_2, float)

    test_output_odsm_61 = wave_smod(61, 10.0)
    assert isinstance(test_output_odsm_61, float)

    test_output_odsm_less_fast_2 = wave_smod(2, 10.0, more_fast = False)
    assert isinstance(test_output_odsm_less_fast_2, float)

    test_output_odsm_less_fast_61 = wave_smod(61, 10.0, more_fast = False)
    assert isinstance(test_output_odsm_less_fast_61, float)
    
    test_output_odmm = wave_mmod(2, 10.0)
    assert isinstance(test_output_odmm, np.ndarray)
    
    test_output_mdsm_2 = wave_smmd(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_mdsm_2, np.ndarray)

    test_output_mdsm_61 = wave_smmd(61, np.array([10.0, 4.5]))
    assert isinstance(test_output_mdsm_61, np.ndarray)

    test_output_mdsm_less_fast_2 = wave_smmd(2, np.array([10.0, 4.5]), more_fast = False)
    assert isinstance(test_output_mdsm_less_fast_2, np.ndarray)

    test_output_mdsm_less_fast_61 = wave_smmd(61, np.array([10.0, 4.5]), more_fast = False)
    assert isinstance(test_output_mdsm_less_fast_61, np.ndarray)
    
    test_output_mdmm = wave_mmmd(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_mdmm, np.ndarray)
    
    test_output_c_odsm_2 = c_wave_smod(2, 10.0 + 0.0j)
    assert isinstance(test_output_c_odsm_2, complex)

    test_output_c_odsm_61 = c_wave_smod(61, 10.0 + 0.0j)
    assert isinstance(test_output_c_odsm_61, complex)

    test_output_c_odsm_less_fast_2 = c_wave_smod(2, 10.0 + 0.0j, more_fast = False)
    assert isinstance(test_output_c_odsm_less_fast_2, complex)

    test_output_c_odsm_less_fast_61 = c_wave_smod(61, 10.0 + 0.0j, more_fast = False)
    assert isinstance(test_output_c_odsm_less_fast_61, complex)
    
    test_output_c_odmm = c_wave_mmod(2, 10.0 + 0.0j)
    assert isinstance(test_output_c_odmm, np.ndarray)
    
    test_output_c_mdsm_2 = c_wave_smmd(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_c_mdsm_2, np.ndarray)

    test_output_c_mdsm_61 = c_wave_smmd(61, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_c_mdsm_61, np.ndarray)

    test_output_c_mdsm_less_fast_2 = c_wave_smmd(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), more_fast = False)
    assert isinstance(test_output_c_mdsm_less_fast_2, np.ndarray)

    test_output_c_mdsm_less_fast_61 = c_wave_smmd(61, np.array([10.0 + 0.0j, 4.5 + 0.0j]), more_fast = False)
    assert isinstance(test_output_c_mdsm_less_fast_61, np.ndarray)
    
    test_output_c_mdmm = c_wave_mmmd(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_c_mdmm, np.ndarray)
    
    print("All functionality tests passed.")

