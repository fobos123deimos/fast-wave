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

from mpmath import mp, matrix
import mpmath
import numpy as np


def psi_n_single_fock_single_position(n: np.uint64, x: np.float64, prec: np.uint64) -> mpmath.ctx_mp_python.mpf:

    """
    Calculates the nth wavefunction to an real scalar x with arbitrary precision using mpmath.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.float64
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.ctx_mp_python.mpf
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_single_position(0,1.0,60)
    mpf('0.45558067201133253483370525689785138607662639040929439687915331')
    >>> psi_n_single_fock_single_position(61,1.0,60)
    mpf('-0.239304919917113097789996116536717211865611421191819349290628243')
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """

    mp.dps = prec
    x = mp.mpf(str(x))
    n = mp.mpf(str(n))
    return  mp.hermite(n, x) * mp.power(mp.mpf('2.0'),(mp.mpf('-0.5') * n)) * mp.power(mp.gamma(n+1.0), mp.mpf('-0.5')) * mp.power(mp.pi , mp.mpf('-0.25')) * mp.exp(-mp.power(x , mp.mpf('2.0')) / mp.mpf('2.0')) 





def psi_n_single_fock_single_position_complex(n: np.uint64, x: np.complex128, prec: np.uint64) -> mpmath.ctx_mp_python.mpc:

    """
    Calculates the nth wavefunction to a complex scalar x with arbitrary precision using mpmath.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.complex128
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.ctx_mp_python.mpc
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_single_position_complex(0,1.0+2.0j,60)
    mpc(real='-1.40087973302624535996319358379185603705205815719366827159881527', imag='-3.06097806029750039193292973729038840279841978760336147713769087')
    >>> psi_n_single_fock_single_position_complex(61,1.0+2.0j,60)
    mpc(real='-511062135.475553070892329856229109412939170026007243421420322129', imag='131445997.757536932748911867174534983962121585813389430606204944')
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """

    mp.dps = prec
    x = mp.mpc(str(x.real),str(x.imag))
    n = mp.mpf(str(n))
    return  mp.hermite(n, x) * mp.power(mp.mpf('2.0'),(mp.mpf('-0.5') * n)) * mp.power(mp.gamma(n+1.0), mp.mpf('-0.5')) * mp.power(mp.pi , mp.mpf('-0.25')) * mp.exp(-mp.power(x , mp.mpf('2.0')) / mp.mpf('2.0')) 






def psi_n_single_fock_multiple_position(n: np.uint64, X: np.ndarray[np.float64], prec: np.uint64) -> mpmath.matrices.matrices._matrix:


    """
    Calculates the nth wavefunction to a real vector x with arbitrary precision using mpmath.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.float64]
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.matrices.matrices._matrix
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_multiple_position(0,np.array([1.0,2.0]),20)
    matrix(
    [['0.45558067201133253483', '0.10165378830641791152']])
    >>> psi_n_single_fock_multiple_position(61,np.array([1.0,2.0]),20)
    matrix(
    [['-0.23930491991711309779', '-0.016773782204892582343']])
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """


    return  matrix([psi_n_single_fock_single_position(n, x, prec) for x in X]).T






def psi_n_single_fock_multiple_position_complex(n: np.uint64, X: np.ndarray[np.complex128], prec: np.uint64) -> mpmath.matrices.matrices._matrix:

    """
    Calculates the nth wavefunction to a complex vector x with arbitrary precision using mpmath.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.complex128]
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.matrices.matrices._matrix
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> psi_n_single_fock_multiple_position_complex(0,np.array([1.0 + 1.0j, 2.0 + 2.0j]),20)
    matrix(
    [[mpc(real='0.40583486367087033308603', imag='-0.63205035161528260798606'), mpc(real='-0.49096842060721693717778', imag='0.56845368634059468652777')]])
    >>> psi_n_single_fock_multiple_position_complex(61,np.array([1.0 + 1.0j, 2.0 + 2.0j]),20)
    matrix(
    [[mpc(real='-7565.4894098859360141926', imag='921.4986211518276840917'), mpc(real='-164189541.53192908120809', imag='-370892077.23796911662203')]])
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """

    return  matrix([psi_n_single_fock_single_position_complex(n, x, prec) for x in X]).T





def psi_n_multiple_fock_single_position(n: np.uint64, x: np.float64, prec: np.uint64) -> mpmath.matrices.matrices._matrix:

    """
    Determines the wavefunction for a real scalar x to all fock states until n, employing mpmath for arbitrary-precision calculations.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.float64
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.matrices.matrices._matrix
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> psi_n_multiple_fock_single_position(1,1.0,60)
    matrix(
    [['0.455580672011332534833705256897851386076626390409294396879153', '0.644288365113475181510837645362740498634994248687269122618738']])
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """


    return  matrix([psi_n_single_fock_single_position(i, x, prec) for i in range(n+1)]).T





def psi_n_multiple_fock_single_position_complex(n: np.uint64, x: np.complex128, prec: np.uint64) -> mpmath.matrices.matrices._matrix:

    """
    Determines the wavefunction for a complex scalar x to all fock states until n, employing mpmath for arbitrary-precision calculations.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.complex128
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.matrices.matrices._matrix
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> c_wavefunction_mmod_arb_prec(1,1.0 +2.0j,20)
    matrix(
    [[mpc(real='-1.400879733026245359964', imag='-3.0609780602975003919354'), mpc(real='6.6766102562991123531695', imag='-8.2911629223978481324862')]])
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """

    return  matrix([psi_n_single_fock_single_position_complex(i, x, prec) for i in range(n+1)]).T





def psi_n_multiple_fock_multiple_position(n: np.uint64, X: np.ndarray[np.float64], prec: np.uint64) -> mpmath.matrices.matrices._matrix:   

    """
    Determines the wavefunction for a real vector x to all fock states until n, employing mpmath for arbitrary-precision calculations.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.float64]
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.matrices.matrices._matrix
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> wavefunction_mmmd_arb_prec(1,np.array([1.0,2.0]),20)
    matrix(
    [['0.45558067201133253483', '0.10165378830641791152'],
    ['0.64428836511347518151', '0.28752033217907949445']])
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """

    return  matrix([[psi_n_single_fock_single_position(i, x, prec) for x in X] for i in range(n+1)])





def psi_n_multiple_fock_multiple_position_complex(n: np.uint64, X: np.ndarray[np.complex128], prec: np.uint64) -> mpmath.matrices.matrices._matrix:   

    """
    Determines the wavefunction for a complex vector x to all fock states until n, employing mpmath for arbitrary-precision calculations.

    Parameters:
    ----------
    n : np.uint64
        Quantum state number.
    x : np.ndarray[np.float128]
        Position(s) at which to evaluate the wavefunction.
    prec : np.uint64
        Desired precision for the calculation (number of decimal digits).

        
    Returns:
    -------
        mpmath.matrices.matrices._matrix
        The evaluated wavefunction with the specified precision.

    Examples
    --------
    ```python
    >>> c_wavefunction_mmmd_arb_prec(1,np.array([1.0+1.0j,2.0+2.0j]),20)
    [[mpc(real='0.40583486367087033308603', imag='-0.63205035161528260798606'), mpc(real='-0.49096842060721693717778', imag='0.56845368634059468652777')],
    [mpc(real='1.4677913476441970351171', imag='-0.31991701106983521979673'), mpc(real='-2.9964982238469495343176', imag='0.21916142736845211639935')]])
    ```

    References:
    ----------
    - The mpmath development team. (2023). mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0). Retrieved from http://mpmath.org/
    """

    return  matrix([[psi_n_single_fock_single_position_complex(i, x, prec) for x in X] for i in range(n+1)])
