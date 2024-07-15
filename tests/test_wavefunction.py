import pytest
import numpy as np
from sympy import symbols
from fast_wave.wavefunction import *

@pytest.fixture(scope="module", autouse=True)
def initialize_c_matrix():
    """
    Fixture to initialize the global variable c_matrix before running tests.
    """
    global c_matrix
    c_matrix = create_hermite_coefficients_matrix(60)

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
    Tests the create_hermite_coefficients_table function to verify if the coefficient matrix is correct.
    """
    n_max = 2
    coeffs_table = create_hermite_coefficients_matrix(n_max)
    
    expected_table = np.zeros((3, 3))
    expected_table[0, 2] = 1  # H0 = 1
    expected_table[1, 1] = 2  # H1 = 2x
    expected_table[2, 0] = 4  # H2 = 4x^2 - 2
    expected_table[2, 2] = -2

    assert np.allclose(coeffs_table, expected_table)

def test_wavefunction_computation():
    """
    Tests the basic functionality of all wavefunction functions.
    """
    # Testing basic functionality
    test_output_udsm = wavefunction_smod(2, 10.0)
    assert isinstance(test_output_udsm, float)
    
    test_output_udmm = wavefunction_mmod(2, 10.0)
    assert isinstance(test_output_udmm, np.ndarray)
    
    test_output_mdsm = wavefunction_smmd(2, (10.0, 4.5))
    assert isinstance(test_output_mdsm, np.ndarray)
    
    test_output_mdmm = wavefunction_mmmd(2, (10.0, 4.5))
    assert isinstance(test_output_mdmm, np.ndarray)
    
    test_output_c_udsm = c_wavefunction_smod(2, 10.0 + 0.0j)
    assert isinstance(test_output_c_udsm, complex)
    
    test_output_c_udmm = c_wavefunction_mmod(2, 10.0 + 0.0j)
    assert isinstance(test_output_c_udmm, np.ndarray)
    
    test_output_c_mdsm = c_wavefunction_smmd(2, (10.0 + 0.0j, 4.5 + 0.0j))
    assert isinstance(test_output_c_mdsm, np.ndarray)
    
    test_output_c_mdmm = c_wavefunction_mmmd(2, (10.0 + 0.0j, 4.5 + 0.0j))
    assert isinstance(test_output_c_mdmm, np.ndarray)
    
    print("All functionality tests passed.")

