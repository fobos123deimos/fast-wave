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
    test_output_odsm = wave_smod(2, 10.0)
    assert isinstance(test_output_odsm, float)

    test_output_odsm_less_fast = wave_smod(2, 10.0, more_fast = False)
    assert isinstance(test_output_odsm_less_fast, float)
    
    test_output_odmm = wave_mmod(2, 10.0)
    assert isinstance(test_output_odmm, np.ndarray)
    
    test_output_mdsm = wave_smmd(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_mdsm, np.ndarray)

    test_output_mdsm_less_fast = wave_smmd(2, np.array([10.0, 4.5]), more_fast = False)
    assert isinstance(test_output_mdsm_less_fast, np.ndarray)
    
    test_output_mdmm = wave_mmmd(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_mdmm, np.ndarray)
    
    test_output_c_odsm = c_wave_smod(2, 10.0 + 0.0j)
    assert isinstance(test_output_c_odsm, complex)

    test_output_c_odsm_less_fast = c_wave_smod(2, 10.0 + 0.0j, more_fast = False)
    assert isinstance(test_output_c_odsm_less_fast, complex)
    
    test_output_c_odmm = c_wave_mmod(2, 10.0 + 0.0j)
    assert isinstance(test_output_c_odmm, np.ndarray)
    
    test_output_c_mdsm = c_wave_smmd(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_c_mdsm, np.ndarray)

    test_output_c_mdsm_less_fast = c_wave_smmd(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), more_fast = False)
    assert isinstance(test_output_c_mdsm_less_fast, np.ndarray)
    
    test_output_c_mdmm = c_wave_mmmd(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_c_mdmm, np.ndarray)
    
    print("All functionality tests passed.")

