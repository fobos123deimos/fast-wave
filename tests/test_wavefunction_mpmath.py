import numpy as np
import mpmath
import src.fast_wave.wavefunction_mpmath as wm

def test_wavefunction_computation():
    """
    Tests the basic functionality of all wavefunction_arb_prec functions.
    """

    # Testing basic functionality
    test_output_sfsp = wm.psi_n_single_fock_single_position(2, 10.0, 20)
    assert isinstance(test_output_sfsp, mpmath.ctx_mp_python.mpf)
    
    test_output_mfsp = wm.psi_n_multiple_fock_single_position(2, 10.0, 20)
    assert isinstance(test_output_mfsp , mpmath.matrices.matrices._matrix)
    
    test_output_sfmp = wm.psi_n_single_fock_multiple_position(2, np.array([10.0, 4.5]), 20)
    assert isinstance(test_output_sfmp, mpmath.matrices.matrices._matrix)
    
    test_output_mfmp = wm.psi_n_multiple_fock_multiple_position(2, np.array([10.0, 4.5]), 20)
    assert isinstance(test_output_mfmp, mpmath.matrices.matrices._matrix)
    
    test_output_sfsp_c = wm.psi_n_single_fock_single_position_complex(2, 10.0 + 0.0j, 20)
    assert isinstance(test_output_sfsp_c, mpmath.ctx_mp_python.mpc)
    
    test_output_mfsp_c = wm.psi_n_multiple_fock_single_position_complex(2, 10.0 + 0.0j, 20)
    assert isinstance(test_output_mfsp_c, mpmath.matrices.matrices._matrix)

    test_output_sfmp_c = wm.psi_n_single_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), 20)
    assert isinstance(test_output_sfmp_c, mpmath.matrices.matrices._matrix)
    
    test_output_mfmp_c = wm.psi_n_multiple_fock_multiple_position_complex(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), 20)
    assert isinstance(test_output_mfmp_c, mpmath.matrices.matrices._matrix)
    
    print("All functionality tests passed.")

