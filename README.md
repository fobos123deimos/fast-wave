

![Fast_Wave_logo](https://github.com/user-attachments/assets/c33a8a1c-96f5-41d4-839c-b687cab2ef01)



![Version](https://img.shields.io/badge/version-1.6.9-blue.svg?cacheSeconds=2592000) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/blob/main/LICENSE)


<br>

> Harnessing the Power of the wavefunctions to navigate the quantum realm.🚀🌌

This project presents an optimized approach for calculating the position wave functions of a Fock state of a quantum harmonic oscillator, with applications in Photonic Quantum Computing simulations. Leveraging [Numba](https://numba.pydata.org/)  [[1](#-references)] and [Cython](https://cython.org/) [[2](#-references)], this approach outperforms the [Mr Mustard](https://mrmustard.readthedocs.io/en/stable/) package [[3, 4](#-references)] in computing a single wave function value at a single position and at multiple positions.

## 📑 Table of Contents

- [📑 Table of Contents](#-table-of-contents)
- [✨ Advantages](#-advantages)
- [🛠️ Setup](#️-setup)
- [🎨 Examples](#-examples)
- [🌊 The Wavefunction](#-the-wavefunction)
  - [Schrödinger Equation](#schrödinger-equation)
  - [Quantum Harmonic Oscillator](#quantum-harmonic-oscillator)
  - [Fock states](#fock-states)
- [🔁 The Wavefunction Recurrence](#-the-wavefunction-recurrence)
- [⚡️The Numba Module - Hybrid Solution](#️the-numba-module---hybrid-solution)
- [⚡️ The Numba Module - Arguments](#️-the-numba-module---arguments)
- [⚙️ The Cython Module](#️-the-cython-module)
- [📖 References](#-references)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📬 Contact](#-contact)

## ✨ Advantages


- **Highly Efficient**: This package includes two fixed-point modules focused on speed. One is implemented using *Numba*, an open-source Just-in-Time (JIT) compiler, and the other module is implemented in *Cython*, a programming language that combines the ease of use of Python with the speed of C.
- **Highly Accurate**: The functions in this package have precision next to the precision of [Wolfram Mathematica and MATLAB](https://drive.google.com/drive/folders/1rdE1l3nBYh1JUuuZNYMFtHd9cVaP-x7g?usp=sharing). In addition, there is a module for calculating wave functions with arbitrary precision using the *mpmath* package.
- **Past response cache**: This package provides a caching module designed to enhance the performance of functions that take multiple positions of a *NumPy* array as input. This module stores previously computed results by leveraging Python's functools.lru_cache, eliminating the need for redundant calculations. This caching module is inspired by the [caching module](https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26) from Mr. Mustard, a package from the photonic quantum computing company Xanadu.


## 🛠️ Setup
To use this package, simply run the following command in the command line: 
```bash
pip install fast-wave
``` 

## 🎨 Examples

The functions `psi_n_multiple_fock_multiple_position` calculate the values of the wavefunction $\psi_{0\rightarrow n}\big(X_{m}\big)$ for multiple Fock states ($n$) and multiple positions ($X_{m}$).

**Inputs:**

* `n`: An integer specifying the maximum Fock state ($n$).
* `X_m`: A 1D `numpy` array with `m` positions, representing the input values where $\psi_{,0\rightarrow n}$ will be evaluated. For example, `np.array([1.0, 2.0])` has dimension $(m,)$, where $m=2$ in this case.

**Outputs:**

* `numpy` **Matrix**: The output has dimensions $(n+1) \times m$, where:
  
  - $n+1$: Corresponds to the Fock states $[0, 1, ..., n]$.
  - $m$: Represents the positions given in `X_m`.

**Demonstration:**

Using the provided inputs:

```python
>>> import fast_wave.wavefunction_numba as wn
Functionality Test Passed: True
>>> import fast_wave.wavefunction_cython as wc
>>> import numpy as np
>>> wn.psi_n_multiple_fock_multiple_position(1,np.array([1.0 ,2.0])) 
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
>>> wc.psi_n_multiple_fock_multiple_position(1,np.array([1.0 ,2.0]))
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
```

**Explanation of the Output:**

* For `n=1` and `X_m = np.array([1.0, 2.0])`:
  - The output matrix has dimensions $(n+1) \times m = 2 \times 2$.
  - The first row contains $\psi_{0}(x_1)$ and $\psi_{0}(x_2)$.
  - The second row contains $\psi_{1}(x_1)$ and $\psi_{1}(x_2)$.

There are other examples in the examples folder: [Speed Tests: Numba & Cython](https://colab.research.google.com/github/fobos123deimos/fast-wave/blob/main/examples/speed_tests_numba_and_cython.ipynb); [Precision Tests: mpmath](https://colab.research.google.com/github/fobos123deimos/fast-wave/blob/main/examples/precision_tests_mpmath.ipynb). In the first one there is a comparison with the [Mr Mustard](https://mrmustard.readthedocs.io/en/stable/) package.

## 🌊 The Wavefunction

The wavefunction, $\Psi(y,t)$, is a fundamental concept in quantum mechanics that describes the quantum state of a particle or system. Its absolute square, $|\Psi(y,t)|^2$, represents the probability density of finding the particle at position $\mathbf{y}$ and time $\mathbf{t}$. Due to the normalization property: $\int_{-\infty}^{\infty} |\Psi(y,t)|^2 dy = 1$ it's guaranteed that for a given time $\mathbf{t}$, the total probability of finding the particle somewhere in space is unity [[5](#-references)].

###  Schrödinger Equation

The wavefunction is the solution to the Schrödinger equation, a fundamental equation in quantum mechanics:

$$
-\Bigg(\frac{\hbar^{2}}{2m}\Bigg) \frac{\partial^2 \Psi(y,t)}{\partial y^{2}} + \Bigg(\frac{m\omega^2 y^2}{2}\Bigg) \Psi(y,t) = \mathbf{i}\hbar \frac{\partial\Psi(y,t)}{\partial t} \quad \mathbf{(1)}
$$

where $\mathbf{\hbar}$ is the reduced Planck constant, $\mathbf{m}$ is the mass of the particle, and $\mathbf{\omega}$ is the angular frequency of the harmonic potential. The symbol $\mathbf{i}$ represents the imaginary unit. When seeking the solution to this equation, we separated the variables as follows: $\Psi(y,t) = \psi(y)f(t)$, and we find as a result for $f(t)$ [[5](#-references)]:

$$ f(t) = C  e^{-iEt/\hbar} \quad \mathbf{(2)}$$

where $\mathbf{C}$ may be considered an arbitrary complex constant and $\mathbf{E}$, the system separation constant can be interpreted as the system's energy. Substituting into the wavefunction we have [[5](#-references)]:

$$ \Psi(y,t) = C  e^{-iEt/\hbar}  \psi(y) \quad \mathbf{(3)}$$

The term $e^{-iEt/\hbar}$ is called the **phase factor** of $\Psi(y,t)$. In order to find $\psi(y)$ we then solve the **time-independent Schröndiger equation** [[5](#-references)]:

$$
-\Bigg(\frac{\hbar^{2}}{2m}\Bigg)  \psi''(y) + \Bigg(\frac{m\omega^2 y^2}{2}\Bigg) \psi(y) = E  \psi(y) \quad \mathbf{(4)}
$$


### Quantum Harmonic Oscillator

By solving equation **(4)**, we obtain a family of energy eigenfunctions defined as follows [[5](#-references)]:

$$
\psi_n(y) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \left(\frac{1}{\sqrt{2^n n!}}\right) H_n\left(\sqrt{\frac{m\omega}{\hbar}}y\right) e^{-m\omega y^2/2\hbar} , \quad  n \in \mathbb{N}_{0} \quad \mathbf{(5)}
$$

where $\mathbf{n}$ represents a non-negative integer corresponding to the different energy states of the system, with energies given by $E_n = \big(n + \frac{1}{2}\big)\hbar \omega$. The term $H_n$ denotes the Hermite polynomial of degree $\mathbf{n}$; thus, for each energy state $\mathbf{n}$, there is an associated Hermite polynomial of degree $\mathbf{n}$ within its eigenfunction [[5](#-references)]:

<br>

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/cd191cda-94cf-47f3-8835-85425e932b22" alt=" " style="max-width: 100%; ">
    <figcaption style="font-style: italic; color: #666; text-align: center;">
    <p style="font-style: italic; color: #666;"> Wavefunctions and energies for different $\mathbf{n}$ values. <a href="#-references">[6]</a></p>
</div>

<br>

The energy eigenfunction for an energy state $\mathbf{n}$ is the wavefunction for an energy state $\mathbf{n}$ of a Quantum Harmonic Oscillator. From this definition, we can then represent the wave function $\Psi(x,t)$ as a series expansion of its family of energy eigenfunctions $\\{\psi_{n}(x)\\}$ [[5](#-references)]:

$$
\Psi(y,t) = \sum_{n=0}^{\infty} c_{n}  \psi_{n}(y)  e^{-\mathbf{i}E_{n}t/\hbar} \quad \mathbf{(6)}
$$

where $\mathbf{c_{n}}$ are complex constants that determine the contribution of each eigenfunction $\psi_{n}(y)$ to the total wavefunction $\Psi(y,t)$. These coefficients are chosen to ensure that the wavefunction satisfies the initial condition of the problem ($t=0$) [[5](#-references)].

### Fock states

When defining the dimensionless variable $x = \Big(m\omega/\hbar\Big)^{1/2}y$, referred to as the **reduced coordinate**, it follows that $dy = \Big(\hbar/m\omega\Big)^{1/2}dx$. As a result, we can write [[7](#-references)]:


$\displaystyle\int_{-\infty}^{+\infty} |\psi(y)|^{2} dy = 1 \implies \int_{-\infty}^{+\infty}\Bigg[\left(\frac{m\omega}{\pi\hbar}\right)^{1/2} \left(\frac{1}{2^n n!}\right) H_n^{2}\left(\sqrt{\frac{m\omega}{\hbar}}y\right) e^{-m\omega y^2/\hbar} \Bigg] dy \implies$ 


$\displaystyle\int_{-\infty}^{+\infty} \Bigg[ \left(\frac{m\omega}{\pi\hbar}\right)^{1/2} \left(\frac{1}{2^n n!}\right) H_{n}^{2}(x)  e^{-x^{2}}\Bigg]\Bigg[\frac{\hbar}{\omega m}\Bigg]^{1/2}dx = 1 \implies \int_{-\infty}^{+\infty} \Bigg[ \left(\frac{1}{2^n n!\sqrt{\pi}}\right) H_{n}^{2}(x)  e^{-x^{2}}\Bigg]dx = 1 \implies$


$\displaystyle\int_{-\infty}^{+\infty}  \left| \left(\frac{1}{2^n n!\sqrt{\pi}}\right)^{1/2} H_{n}(x)  e^{-x^{2}/2} \right|^{2}dx = 1 \implies \int_{-\infty}^{+\infty} |\psi(x)|^{2} dx = 1\implies$

$$\psi_{n}(x) = \left(\frac{1}{2^n n!\sqrt{\pi}}\right)^{1/2} H_{n}(x) \quad e^{-x^{2}/2} \quad \mathbf{(7)}$$

This demonstrates that the wavefunction of a Quantum Harmonic Oscillator can be represented in a more dimensionless form, known as the **Hermite function**, it is also sometimes referred to as the **Gauss-Hermite function**. (equation $\mathbf{7}$) [[8](#-references)]. A more simplified form for this type of function is this: $\psi_{n}(x) = \mathcal{N_{n}}  H_{n}(x)  e^{-x^{2}/2}$, where $\mathcal{N_{n}} = \left[1 /(2^n n!\sqrt{\pi})\right]^{1/2}$ is referred to as the **normalization constant** [[9](#-references)].

This type of representation for the wavefunction enables the modeling of wave functions for **Fock states**, which are energy eigenstates with a well-defined number of particles. When the particles are considered photons, $\mathbf{n}$, the degree of the Hermite polynomial, also represents the number of photons. Consequently, the Fock states ($\ket{n}$) become eigenstates of the photon number operator $\hat{n}$, such that $\hat{n}\ket{n}=n\ket{n}$. Moreover, the energy states now correspond to the energy levels of an optical system containing $n$-photons [[10](#-references)].


$\ket{\psi_{n}} = \displaystyle\frac{1}{\sqrt{n!}}(a^{\dagger})^{n}\ket{\psi_{0}} \implies \psi_{n}(q) = \frac{1}{\sqrt{n!}}\frac{1}{\sqrt{2^{n}}}\Bigg(q - \frac{\partial}{\partial q}\Bigg)^{n}\psi_{0}(q) \implies $

$\psi_{n}(q) = \displaystyle\frac{1}{\sqrt{n!}}\frac{1}{\sqrt{2^{n}}}  H_{n}(q)  \psi_{0}(q) \implies \psi_{n}(q) = \displaystyle\frac{1}{\sqrt{n!}}\frac{1}{\sqrt{2^{n}}}  H_{n}(q) \Big(\pi^{-1/4}\Big) e^{-q^{2}/2} \implies$

$$\psi_{n}(q) = \displaystyle\left(\frac{1}{2^n n!\sqrt{\pi}}\right)^{1/2} H_{n}(q) \quad e^{-q^{2}/2} \quad \mathbf{(8)}$$

where $\mathbf{q}$ is interpreted as the normalized field amplitude associated with the position quadrature ($\hat{q}$​) and has a dimensionless character similar to $\mathbf{x}$ [[10](#-references)]. 

The wavefunction $\mathbf{n}$ of a Fock state is the projection of the state $\ket{n}$, and as we can observe, the wavefunctions of a Fock state for $n$-photon systems can be interpreted as a family of dimensionless energy eigenfunctions (Hermite functions), similar to those shown by equation $\mathbf{(7)}$ [[10](#-references)].

## 🔁 The Wavefunction Recurrence

In essence, Mr Mustard's strategy is to use the [Renormalized Hermite Polynomial](https://mrmustard.readthedocs.io/en/stable/code/api/mrmustard.math.hermite_renormalized.html) [[3, 4](#-references)] for the computation of the wavefunction of a quantum harmonic oscillator. Below, we show the recurrence for calculating the Renormalized Hermite Polynomial, as well as the method for calculating it using the traditional Hermite polynomial:

$$H_{n+1}^{re}(x) = \displaystyle\frac{2}{\sqrt{n+1}}\bigg[xH_{n}^{re}(x) - H_{n-1}^{re}(x)\sqrt{n-1}\bigg] \quad \mathbf{(8)} $$ 

$$H_{n}^{re}(x) = \displaystyle\frac{H_{n}(x)}{\sqrt{n!}} \quad \mathbf{(10)} $$ 

When we use this polynomial in calculating the wavefunction of a Quantum Harmonic Oscillator, the equation is as follows:

$$\psi_{n}(x) = \displaystyle\Bigg(\frac{1}{2^n\sqrt{\pi}}\Bigg)^{1/2}H_{n}^{re}(x) \quad e^{-\frac{x^{2}}{2}} \quad \mathbf{(11)} $$ 

In this package, we implemented a recurrence based on the recursive solution to the wavefunction of the Quantum Harmonic Oscillator presented in the work of *José Maria Pérez-Jordá* [[11](#-references)]. The recurrence we implemented was for $\psi_{n+1}$, which we obtained from the recursive definition of the Hermite polynomial [[12](#-references)], as suggested by *José Maria Pérez-Jordá* in his article:


$H_{n+1}(x) = 2xH_{n}(x) - 2nH_{n-1}(x) \implies $


$\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)H_{n+1}(x) = \Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2xH_{n}(x) -\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2nH_{n-1}(x) \implies$


$\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)H_{n+1}(x) = \Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2xH_{n}(x) -2n\psi_{n-1}(x) \implies $


$\displaystyle\Bigg(\frac{1}{\sqrt{2n}}\Bigg)\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)H_{n+1}(x) = \Bigg(\frac{1}{\sqrt{2n}}\Bigg)\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2xH_{n}(x) -\Bigg(\frac{1}{\sqrt{2n}}\Bigg)2n\psi_{n-1}(x) \implies$


$\Bigg(\displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n}n!\pi^{1/2}}}\Bigg) H_{n+1}(x) = 2x\psi_{n}(x) - \Bigg(\frac{2n}{\sqrt{2n}}\Bigg)\psi_{n-1}(x) \implies$


$\displaystyle\Bigg(\frac{1}{\sqrt{2(n+1)}}\Bigg)\Bigg(\displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n}n!\pi^{1/2}}}\Bigg) H_{n+1}(x) = \displaystyle\Bigg(\frac{1}{\sqrt{2(n+1)}}\Bigg)2x\psi_{n}(x) - \displaystyle\Bigg(\frac{1}{\sqrt{2(n+1)}}\Bigg)\Bigg(\frac{2n}{\sqrt{2n}}\Bigg)\psi_{n-1}(x) \implies$


$$\psi_{n+1}(x) = \displaystyle\Bigg(\sqrt{\frac{2}{n+1}}\Bigg)x\psi_{n}(x) -\Bigg(\sqrt{\frac{n}{n+1}}\Bigg)\psi_{n-1}(x) \quad \mathbf{(12)}$$

Besides the use of this recurrence in this package, the same authors implemented a version of it in a Cython module of QuTip: [_distributions.pyx](https://github.com/qutip/qutip/blob/master/qutip/_distributions.pyx), to be used in the HarmonicOscillatorWaveFunction class from the [distributions.py](https://github.com/qutip/qutip/blob/master/qutip/distributions.py) module.

## ⚡️The Numba Module - Hybrid Solution

We use a hybrid solution with two algorithms for calculating the wave function for calculating a single Fock wave function's values at multiple positions (Single Fock and Multiple Position) (`psi_n_single_fock_multiple_position`). For $n>60$ or more than 35 positions, we use the recurrence for the wave function. For $n\le 60$ and at most 35 positions we use a precomputed matrix with the normalized coefficients of the Hermite polynomial as follows:


$$\psi_{i}(x) = \displaystyle\frac{1}{\sqrt{2^{i}i!\pi^{1/2}}}H_{i}(x)e^{-x^{2}/2} = \frac{1}{\sqrt{2^{i}i!\pi^{1/2}}}\mathbf{C_{n}[i]} \cdot  \mathbf{x^{p}} e^{-x^{2}/2} \implies $$


$$\psi_{i}(x) = \mathbf{C^{s}_{n}[i]\cdot x^{p}e^{-x^{2}/2} \quad \mathbf{(13)}}$$


where $\mathbf{C^{s}_{n}[i]}$ is the row vector of normalized coefficients that multiply each power of $x$ up to $x^n$. The entire matrix $\mathbf{C^s_n}$ of such rows is precomputed up to degree $n=60$.  $\mathbf{x^{p}}$ is a column vector of powers up to n, with zeros in places where the coefficient is zero; for example, for $i=3$, $\mathbf{x^{p}} = [x^{3}, 0.0, x^{1}, 0.0]^T$. This hybrid algorithm is also used in Single Fock and Single Position (`psi_n_single_fock_single_position`) problems, though it offers no computational advantage in these cases. Additionally, there is an argument named **CS_matrix** for these Single Fock functions, set to **True** to enable the use of this matrix. In other words, you can use only the recurrence relation for the wave function at any value. The use of this coefficient matrix is limited to values up to **60** (determined empirically), as beyond this point, the function may encounter precision errors, resulting in incoherent outputs [[13](#-references)].

## ⚡️ The Numba Module - Arguments

For this algorithm to perform as efficiently as possible, Numba's Just-in-Time compilation is used in conjunction with [lru_cache (Least Recently Used - Cache Management)](https://docs.python.org/3/library/functools.html). The following arguments were used in the **@nb.jit** decorator:

- **nopython=True:** This argument forces the Numba compiler to operate in "nopython" mode, which means that all the code within the function must be compilable to pure machine code without falling back to the Python interpreter. This results in significant performance improvements by eliminating the overhead of the Python interpreter.
- **looplift=True:** This argument allows Numba to "lift" loops out of "nopython" mode. That is, if there are loops in the code that cannot be compiled in "nopython" mode, Numba will try to move them outside of the compiled part and execute them as normal Python code.
- **nogil=True:** This argument releases the Python Global Interpreter Lock (GIL) while the function is executing. It is useful for allowing the Numba-compiled code to run in parallel with other Python threads, increasing performance in multi-threaded programs.
- **boundscheck=False:** Disables array bounds checking. Normally, Numba checks if array indices are within valid bounds. Disabling this check can increase performance but may result in undefined behavior if there are out-of-bounds accesses.
- **cache=True:** Enables caching of the compiled function. The first time the function is compiled, Numba stores the compiled version in a cache. On subsequent executions, Numba can reuse the compiled version from the cache instead of recompiling the function, reducing the function's startup time.

## ⚙️ The Cython Module

The Cython module includes compiled files for Linux (**.so**) and Windows (**.pyd**), which allows it to be used in Google Colab (Linux). Additionally, this module supports three versions of Python 3: 3.10, 3.11, and 3.12. All these files are placed in the package folder upon installation. The source code of the Cython module is available in the repository in **.pyx** format. In the functions of the Cython module, some decorators are used to increase speed:

- **@cython.nogil**: This decorator allows a Cython function to release the Global Interpreter Lock (GIL), making it possible to execute that block of code concurrently in multiple threads.
- **@cython.cfunc**:  This decorator tells Cython to treat the function as a C function, meaning it can be called from other Cython or C code, not just Python. The function will have C-level calling conventions.
- **@cython.locals(...)**: Declares local variable types to optimize performance.
- **@cython.boundscheck(False)**: Disables bounds checking for arrays/lists to boost speed, but at the cost of safety. 


## 📖 References

Our journey through the quantum realm is inspired by the following:

   1. Lam, S. K., Pitrou, A., & Seibert, S. (2015). _Numba: A LLVM-based Python JIT compiler_. In _Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC_ (LLVM '15) (pp. 7-12). Association for Computing Machinery. https://doi.org/10.1145/2833157.2833162
   2. Behnel, S., Bradshaw, R., Citro, C., Dalcin, L., Seljebotn, D. S., & Smith, K. (2011). *Cython: The best of both worlds*. Computing in Science & Engineering, 13(2), 31-39. https://doi.org/10.1109/MCSE.2010.118
   3. Yao, Y., Miatto, F., & Quesada, N. (2024). _Riemannian optimization of photonic quantum circuits in phase and Fock space_ [Preprint]. arXiv:2209.06069. [https://doi.org/10.21468/SciPostPhys.17.3.082](https://doi.org/10.21468/SciPostPhys.17.3.082)
   4. Miatto, F. M., & Quesada, N. (2020). *_Fast optimization of parametrized quantum optical circuits_* (*Quantum*, 4, 366). [https://doi.org/10.22331/q-2020-11-30-366](https://doi.org/10.22331/q-2020-11-30-366)
   5. Bowers, P. L. (2020). *Lectures on Quantum Mechanics: A Primer for Mathematicians*. Cambridge University Press. ISBN: [1108429769](https://www.worldcat.org/isbn/1108429769) ([9781108429764](https://www.worldcat.org/isbn/9781108429764))
   6. Aerts, D., Beltran, L. *Quantum Structure in Cognition: Human Language as a Boson Gas of Entangled Words*. Found Sci 25, 755–802 (2020). [https://doi.org/10.1007/s10699-019-09633-4](https://doi.org/10.1007/s10699-019-09633-4)
   7. Beiser, A. (2003). *Concepts of Modern Physics*. 6th ed. McGraw Hill. ISBN: [0072448482](https://www.worldcat.org/isbn/0072448482) ([9780072448481](https://www.worldcat.org/isbn/9780072448481))
   8. Celeghini, E., Gadella, M., & del Olmo, M. A. (2021). *Hermite functions and Fourier series*. Symmetry, 13(5), Article 853. [https://doi.org/10.3390/sym13050853](https://doi.org/10.3390/sym13050853)
   9. Schleich, W. P. (2001). *Quantum optics in phase space*. Wiley-VCH. ISBN: [352729435X](https://www.worldcat.org/isbn/352729435X) ([9783527294350](https://www.worldcat.org/isbn/9783527294350))
   10. Leonhardt, U. (2010). Essential Quantum Optics: From Quantum Measurements to Black Holes. Cambridge: Cambridge University Press. ISBN: [0521869781](https://www.worldcat.org/isbn/0521869781) ([9780521869782](https://www.worldcat.org/isbn/9780521869782))
   11. Pérez-Jordá, J. M. (2017). *On the recursive solution of the quantum harmonic oscillator*. European Journal of Physics, 39(1), 015402. [https://doi.org/10.1088/1361-6404/aa9584](https://doi.org/10.1088/1361-6404/aa9584)
   12. Olver, F. W. J., & Maximon, L. C. (2010). *NIST Handbook of Mathematical Functions*. Cambridge University Press. ISBN: [0521192250](https://www.worldcat.org/isbn/0521192250) ([9780521192255](https://www.worldcat.org/isbn/9780521192255))
   13. Cordeiro, M., Bezerra, I. P., & Vasconcelos, H. H. M. (2024). *Efficient computation of the wave function ψn(x) using Hermite coefficient matrix in Python*. In 7º Workshop Escola de Computação e Informação Quântica (7ª WECIQ) (pp. 56-60). CEFET/RJ.

## 🤝 Contributing

Contributions, whether filing an issue, proposing improvements, or submitting a pull request, are welcome! Please feel free to explore, ask questions, and share your ideas.

## 📜 License

This project is available under the *BSD 3-Clause License*. See the LICENSE file for more details.

## 📬 Contact

If you have any questions or want to reach out to the team, please send me an email at [matheusgomescord@gmail.com](matheusgomescord@gmail.com).
