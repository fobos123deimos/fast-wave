import numpy as np
import src.fast_wave.wavefunction_cython as wc

def test_wavefunction_computation():
    """
    Tests the basic functionality of all wavefunction functions.
    """

    # Testing basic functionality
    test_output_sfsp_2 = wc.psi_n_single_fock_single_position(2, 10.0)
    assert isinstance(test_output_sfsp_2, float)

    test_output_sfsp_61 = wc.psi_n_single_fock_single_position(61, 10.0)
    assert isinstance(test_output_sfsp_61, float)
    
    test_output_mfsp = wc.psi_n_multiple_fock_single_position(2, 10.0)
    assert isinstance(test_output_mfsp , np.ndarray)
    
    test_output_sfmp_2 = wc.psi_n_single_fock_multiple_position(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_sfmp_2, np.ndarray)

    test_output_sfmp_61 = wc.psi_n_single_fock_multiple_position(61, np.array([10.0, 4.5]))
    assert isinstance(test_output_sfmp_61, np.ndarray)

    test_output_mfmp = wc.psi_n_multiple_fock_multiple_position(2, np.array([10.0, 4.5]))
    assert isinstance(test_output_mfmp, np.ndarray)
    
    test_output_sfsp_c_2 = wc.psi_n_single_fock_single_position_complex(2, 10.0 + 0.0j)
    assert isinstance(test_output_sfsp_c_2, complex)

    test_output_sfsp_c_61 = wc.psi_n_single_fock_single_position_complex(61, 10.0 + 0.0j)
    assert isinstance(test_output_sfsp_c_61, complex)
    
    test_output_mfsp_c = wc.psi_n_multiple_fock_single_position_complex(2, 10.0 + 0.0j)
    assert isinstance(test_output_mfsp_c, np.ndarray)
    
    test_output_sfmp_c_2 = wc.psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_sfmp_c_2, np.ndarray)

    test_output_sfmp_c_61 = wc.psi_n_single_fock_multiple_position_complex(61, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_sfmp_c_61, np.ndarray)

    test_output_sfmp_c_less_fast_2 = wc.psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_sfmp_c_less_fast_2, np.ndarray)

    test_output_sfmp_c_less_fast_61 = wc.psi_n_single_fock_multiple_position_complex(61, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_sfmp_c_less_fast_61, np.ndarray)
    
    test_output_mfmp_c = wc.psi_n_multiple_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]))
    assert isinstance(test_output_mfmp_c, np.ndarray)
    
    print("All functionality tests passed.")