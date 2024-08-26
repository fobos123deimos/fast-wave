import numpy as np
from src.fast_wave.wavefunction_arb_prec import *

def test_wavefunction_computation():
    """
    Tests the basic functionality of all wavefunction_arb_prec functions.
    """

    wave_smod_ap = wavefunction_arb_prec(s_mode = True, o_dimensional = True, complex_bool = False, cache = False, cache_size = 128)
    wave_smmd_ap = wavefunction_arb_prec(s_mode = True, o_dimensional = False, complex_bool = False, cache = False, cache_size = 128)
    wave_mmod_ap = wavefunction_arb_prec(s_mode = False, o_dimensional = True, complex_bool = False, cache = False, cache_size = 128)
    wave_mmmd_ap = wavefunction_arb_prec(s_mode = False, o_dimensional = False, complex_bool = False, cache = False, cache_size = 128)
    c_wave_smod_ap = wavefunction_arb_prec(s_mode = True, o_dimensional = True, complex_bool = True, cache = False, cache_size = 128)
    c_wave_smmd_ap = wavefunction_arb_prec(s_mode = True, o_dimensional = False, complex_bool = True, cache = False, cache_size = 128)
    c_wave_mmod_ap = wavefunction_arb_prec(s_mode = False, o_dimensional = True, complex_bool = True, cache = False, cache_size = 128)
    c_wave_mmmd_ap = wavefunction_arb_prec(s_mode = False, o_dimensional = False, complex_bool = True, cache = False, cache_size = 128)

    # Testing basic functionality
    test_output_odsm = wave_smod_ap(2, 10.0, 20)
    assert isinstance(test_output_odsm, mpmath.ctx_mp_python.mpf)
    
    test_output_odmm = wave_mmod_ap(2, 10.0, 20)
    assert isinstance(test_output_odmm, mpmath.matrices.matrices._matrix)
    
    test_output_mdsm = wave_smmd_ap(2, np.array([10.0, 4.5]), 20)
    assert isinstance(test_output_mdsm, mpmath.matrices.matrices._matrix)
    
    test_output_mdmm = wave_mmmd_ap(2, np.array([10.0, 4.5]), 20)
    assert isinstance(test_output_mdmm, mpmath.matrices.matrices._matrix)
    
    test_output_c_odsm = c_wave_smod_ap(2, 10.0 + 0.0j, 20)
    assert isinstance(test_output_c_odsm, mpmath.ctx_mp_python.mpc)
    
    test_output_c_odmm = c_wave_mmod_ap(2, 10.0 + 0.0j, 20)
    assert isinstance(test_output_c_odmm, mpmath.matrices.matrices._matrix)

    test_output_c_mdsm = c_wave_smmd_ap(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), 20)
    assert isinstance(test_output_c_mdsm, mpmath.matrices.matrices._matrix)
    
    test_output_c_mdmm = c_wave_mmmd_ap(2, np.array([10.0 + 0.0j, 4.5 + 0.0j]), 20)
    assert isinstance(test_output_c_mdmm, mpmath.matrices.matrices._matrix)
    
    print("All functionality tests passed.")

