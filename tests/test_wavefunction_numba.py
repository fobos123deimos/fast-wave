import pytest
import numpy as np
from sympy import symbols
import src.fast_wave.wavefunction_numba as wn

@pytest.fixture(scope="module", autouse=True)
def initialize_c_s_matrix():
    """
    Fixture to initialize the global variable c_s_matrix before running tests.
    """
    global c_s_matrix
    c_s_matrix = wn.create_normalized_hermite_coefficients_matrix(60)

def test_hermite_sympy():
    """
    Tests the hermite_sympy function to verify the accuracy of Hermite polynomial computation.
    """
    x =  symbols("x")
    h0 = wn.hermite_sympy(0)
    h1 = wn.hermite_sympy(1)
    h2 = wn.hermite_sympy(2)

    assert h0 == 1
    assert h1 == 2 * x
    assert h2 == 4 * x**2 - 2

def test_create_hermite_coefficients_table():
    """
    Tests the create_normalized_hermite_coefficients_table function to verify if the normalized coefficient matrix is correct.
    """
    n_max = 2
    coeffs_table = wn.create_normalized_hermite_coefficients_matrix(n_max)
    
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

    # Testing basic functionality
    test_output_sfsp_2 = wn.psi_n_single_fock_single_position(2, 10.0)
    assert isinstance(test_output_sfsp_2, float)

    test_output_sfsp_61 = wn.psi_n_single_fock_single_position(61, 10.0)
    assert isinstance(test_output_sfsp_61, float)

    test_output_sfsp_less_fast_2 = wn.psi_n_single_fock_single_position(2, 10.0, more_fast = False)
    assert isinstance(test_output_sfsp_less_fast_2, float)

    test_output_sfsp_less_fast_61 = wn.psi_n_single_fock_single_position(61, 10.0, more_fast = False)
    assert isinstance(test_output_sfsp_less_fast_61, float)
    
    test_output_mfsp = wn.psi_n_multiple_fock_single_position(2, 10.0)
    assert isinstance(test_output_mfsp , np.ndarray)
    
    test_output_sfmp_2 = wn.psi_n_single_fock_multiple_position(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_sfmp_2, np.ndarray)

    test_output_sfmp_61 = wn.psi_n_single_fock_multiple_position(61, np.array([10.0, 4.5]))
    assert isinstance(test_output_sfmp_61, np.ndarray)

    test_output_sfmp_less_fast_2 = wn.psi_n_single_fock_multiple_position(2, np.array([10.0, 4.5]), more_fast = False)
    assert isinstance( test_output_sfmp_less_fast_2, np.ndarray)

    test_output_sfmp_less_fast_61 = wn.psi_n_single_fock_multiple_position(61, np.array([10.0, 4.5]), more_fast = False)
    assert isinstance(test_output_sfmp_less_fast_61, np.ndarray)
    
    test_output_mfmp = wn.psi_n_multiple_fock_multiple_position(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_mfmp, np.ndarray)
    
    test_output_sfsp_c_2 = wn.psi_n_single_fock_single_position_complex(2, 10.0 + 0.0j)
    assert isinstance(test_output_sfsp_c_2, complex)

    test_output_sfsp_c_61 = wn.psi_n_single_fock_single_position_complex(61, 10.0 + 0.0j)
    assert isinstance(test_output_sfsp_c_61, complex)

    test_output_sfsp_c_less_fast_2 = wn.psi_n_single_fock_single_position_complex(2, 10.0 + 0.0j, more_fast = False)
    assert isinstance(test_output_sfsp_c_less_fast_2, complex)

    test_output_sfsp_c_less_fast_61 = wn.psi_n_single_fock_single_position_complex(61, 10.0 + 0.0j, more_fast = False)
    assert isinstance(test_output_sfsp_c_less_fast_61, complex)
    
    test_output_mfsp_c = wn.psi_n_multiple_fock_single_position_complex(2, 10.0 + 0.0j)
    assert isinstance(test_output_mfsp_c, np.ndarray)
    
    test_output_sfmp_c_2 = wn.psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_sfmp_c_2, np.ndarray)

    test_output_sfmp_c_61 = wn.psi_n_single_fock_multiple_position_complex(61, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_sfmp_c_61, np.ndarray)

    test_output_sfmp_c_less_fast_2 = wn.psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), more_fast = False)
    assert isinstance(test_output_sfmp_c_less_fast_2, np.ndarray)

    test_output_sfmp_c_less_fast_61 = wn.psi_n_single_fock_multiple_position_complex(61, np.array([10.0 + 0.0j, 4.5 + 0.0j]), more_fast = False)
    assert isinstance(test_output_sfmp_c_less_fast_61, np.ndarray)
    
    test_output_mfmp_c = wn.psi_n_multiple_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_mfmp_c, np.ndarray)
    
    print("All functionality tests passed.")

