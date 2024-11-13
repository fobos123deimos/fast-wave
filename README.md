


![Fast_Wave_logo](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/e1de91d2-3792-4b21-9553-7c13ce372a76)


![Version](https://img.shields.io/badge/version-1.5.1-blue.svg?cacheSeconds=2592000) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/blob/main/LICENSE)


<br>

> Harnessing the Power of the wavefunctions to navigate the quantum realm.🚀🌌

This project presents an optimized approach for calculating the position wave functions of a Fock state of a quantum harmonic oscillator, with applications in Photonic Quantum Computing simulations. Leveraging [Numba](https://numba.pydata.org/)  [[1](#-ref)] and [Cython](https://cython.org/) [[2](#-ref)], this approach outperforms the [Mr Mustard](https://mrmustard.readthedocs.io/en/stable/) package [[3,4](#-ref)] in computing a single wave function value at a single position and at multiple positions.

## 📑 Table of Contents

- [📑 Table of Contents](#-table-of-contents)
- [✨ Advantages](#-advantages)
- [🛠️ Setup](#️-setup)
- [🎨 Examples](#-examples)
- [🌊 The Wavefunction](#-the-wavefunction)
  - [Schrödinger Equation](#schrödinger-equation)
  - [Normalization](#normalization)
  - [Quantum Harmonic Oscillator](#quantum-harmonic-oscillator)
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
- **Highly Accurate**: The functions in this package have precision next to the precision of Wolfram Mathematica and MATLAB. In addition, there is a module for calculating wave functions with arbitrary precision using the *mpmath* package.
- **Past response cache**: This package provides a caching module designed to enhance the performance of functions that take multiple positions of a *NumPy* array as input. This module stores previously computed results by leveraging Python's functools.lru_cache, eliminating the need for redundant calculations. This caching module is inspired by the [caching module](https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26) from Mr. Mustard, a package from the photonic quantum computing company Xanadu.


## 🛠️ Setup
To use this package, simply run the following command in the command line: 
```bash
pip install fast-wave
``` 

## 🎨 Examples

```python
>>> import fast_wave.wavefunction_numba as wn
Functionality Test Passed: True
>>> import fast_wave.wavefunction_cython as wc
>>> import numpy as np
>>> wn.psi_n_multiple_fock_multiple_position(1,np.array([1.0 ,2.0]))
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
       [Scott: why did that make a 2x2 array?]
>>> wc.psi_n_multiple_fock_multiple_position(1,np.array([1.0 ,2.0]))
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
```

There are other examples in the examples folder: [Speed Tests: Numba & Cython](https://colab.research.google.com/github/fobos123deimos/fast-wave/blob/main/examples/speed_tests_numba_and_cython.ipynb); [Precision Tests: mpmath](https://colab.research.google.com/github/fobos123deimos/fast-wave/blob/main/examples/precision_tests_mpmath.ipynb). In the first one there is a comparison with the [Mr Mustard](https://mrmustard.readthedocs.io/en/stable/) package.

## 🌊 The Wavefunction

The wavefunction, $\psi(x)$, is a fundamental concept in quantum mechanics that describes the quantum state of a particle or system. The absolute square of the wavefunction, $|\psi(x)|^2$, represents the probability density of finding the particle at a position $x$ [[5](#-ref)].

### Schrödinger Equation

The behavior of a wavefunction is governed by the Schrödinger equation, a fundamental equation in quantum mechanics [[2](#-references)]:

$$
i\hbar\frac{\partial}{\partial t}\Psi(x,t) = \Bigg[-\frac{\hbar}{2m}\frac{\partial^2}{\partial x^{2}} + V(x,t)\Bigg]\Psi(x,t)
$$

where $i$ represents the imaginary unit, and $hbar$ is the reduced Planck constant. The parameter $m$ denotes the mass of the particle, while $V(x,t)$ represents the potential that defines the environment in which the particle exists. The expression $\big[-\frac{\hbar}{2m}\frac{\partial^2}{\partial x^{2}} + V(x,t)\big]$ is the Hamiltonian operator $\hat{H}$. The function $\Psi(x,t)$, called the wavefunction, describes the state of the system at position $x$ and time $t$ and yields complex values.

### Normalization

For the wavefunction to be physically meaningful, it must be normalized [[6](#-ref)]:

$$
\int_{-\infty}^{\infty} |\psi(x,t)|^2 dx = 1
$$

This ensures that the total probability of finding the particle somewhere in space is one.

### Quantum Harmonic Oscillator

The $n$-Fock state wave function of the quantum harmonic oscillator, a system that models particles in a quadratic potential well, are given by [[7](#-ref)]:

$$
\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n\left(\sqrt{\frac{m\omega}{\hbar}}x\right) e^{-\frac{m\omega x^2}{2\hbar}}
$$

where $n$ is a non-negative integer, $m$ is the mass of the particle, $\omega$ is the angular frequency of the oscillator, and $H_n$ are the Hermite polynomials. 

## 🔁 The Wavefunction Recurrence

In essence, Mr Mustard's strategy is to use the [Renormalized Hermite Polynomial](https://mrmustard.readthedocs.io/en/stable/code/api/mrmustard.math.hermite_renormalized.html) [[3,4](#-ref)] for the computation of the wave function of a quantum harmonic oscillator. Below, we show the recurrence for calculating the Renormalized Hermite Polynomial, as well as the method for calculating it using the traditional Hermite polynomial:

$$H_{n+1}^{\; re}(x) = \displaystyle\frac{2}{\sqrt{n+1}}\bigg[xH_{n}^{\; re}(x) - H_{n-1}^{\; re}(x)\sqrt{n-1}\bigg]$$

$$H_{n}^{re}(x) = \displaystyle\frac{H_{n}(x)}{\sqrt{n!}}$$

When we use this polynomial in calculating the wavefunction of a Quantum Harmonic Oscillator, the equation is as follows:

$$\psi_{n}(x) = \displaystyle\frac{1}{\sqrt{2^n}}H_{n}^{\; re}(x)e^{-\frac{x^{2}}{2}}$$

In this package, we implemented a recurrence based on the recursive solution to the wavefunction of the Quantum Harmonic Oscillator presented in the work of José Maria Pérez-Jordá [[7](#-ref)]. The recurrence we implemented was for $\psi_{n+1}$, which we obtained from the recursive definition of the Hermite polynomial [[9](#-ref)], as suggested by José Maria Pérez-Jordá in his article:


$H_{n+1}(x) = 2xH_{n}(x) - 2nH_{n-1}(x) \implies $


$\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)H_{n+1}(x) = \Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2xH_{n}(x) -\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2nH_{n-1}(x) \implies$


$\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)H_{n+1}(x) = \Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2xH_{n}(x) -2n\psi_{n-1}(x) \implies $


$\displaystyle\Bigg(\frac{1}{\sqrt{2n}}\Bigg)\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)H_{n+1}(x) = \Bigg(\frac{1}{\sqrt{2n}}\Bigg)\Bigg( \displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n-1}(n-1)!\pi^{1/2}}}\Bigg)2xH_{n}(x) -\Bigg(\frac{1}{\sqrt{2n}}\Bigg)2n\psi_{n-1}(x) \implies$


$\Bigg(\displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n}n!\pi^{1/2}}}\Bigg) H_{n+1}(x) = 2x\psi_{n}(x) - \Bigg(\frac{2n}{\sqrt{2n}}\Bigg)\psi_{n-1}(x) \implies$


$\displaystyle\Bigg(\frac{1}{\sqrt{2(n+1)}}\Bigg)\Bigg(\displaystyle\frac{e^{-x^{2}/2}}{\sqrt{2^{n}n!\pi^{1/2}}}\Bigg) H_{n+1}(x) = \displaystyle\Bigg(\frac{1}{\sqrt{2(n+1)}}\Bigg)2x\psi_{n}(x) - \displaystyle\Bigg(\frac{1}{\sqrt{2(n+1)}}\Bigg)\Bigg(\frac{2n}{\sqrt{2n}}\Bigg)\psi_{n-1}(x) \implies$


$\psi_{n+1}(x) = \displaystyle\Bigg(\sqrt{\frac{2}{n+1}}\Bigg)x\psi_{n}(x) -\Bigg(\sqrt{\frac{n}{n+1}}\Bigg)\psi_{n-1}(x) \quad \centerdot$




## ⚡️The Numba Module - Hybrid Solution

We use a hybrid solution with two algorithms for calculating the wave function for calculating a single Fock wave function's values at multiple positions (Single Fock and Multiple Position) (`psi_n_single_fock_multiple_position`). For $n>60$ or more than 35 positions, we use the recurrence for the wave function. For $n\le 60$ and at most 35 positions we use a precomputed matrix with the normalized coefficients of the Hermite polynomial as follows:


$$\psi_{i}(x) = \displaystyle\frac{1}{\sqrt{2^{i}i!\pi^{1/2}}}H_{i}(x)e^{-x^{2}/2} = \frac{1}{\sqrt{2^{i}i!\pi^{1/2}}}\mathbf{C_{n}[i]} \cdot  \mathbf{x^{p}} e^{-x^{2}/2} \implies $$


$$\psi_{i}(x) = \mathbf{C^{s}_{n}[i]\cdot x^{p}e^{-x^{2}/2} \quad \centerdot}$$


In this equation, $\mathbf{C^{s}_{n}[i]}$ is the row vector of normalized coefficients that multiply each power of $x$ up to $x^n$. The entire matrix $\mathbf{C^s_n}$ of such rows is precomputed up to degree $n=60$[Scott: is that true?].  $\mathbf{x^{p}}$ is a column vector of powers up to n, with zeros in places where the coefficient is zero; for example, for $i=3$, $\mathbf{x^{p}} = [x^{3}, 0.0, x^{1}, 0.0]^T$. This hybrid algorithm is also used in Single Fock and Single Position (`psi_n_single_fock_single_position`) problems, though it offers no computational advantage in these cases. Additionally, there is an argument named **CS_matrix** for these Single Fock functions, set to **True** to enable the use of this matrix. In other words, you can use only the recurrence relation for the wave function at any value. The use of this coefficient matrix is limited to values up to **60** (determined empirically), as beyond this point, the function may encounter precision errors, resulting in incoherent outputs [[10](#-ref)].

## ⚡️ The Numba Module - Arguments

For this algorithm to perform as efficiently as possible, Numba's Just-in-Time compilationis used in conjunction with [lru_cache (Least Recently Used - Cache Management)](https://docs.python.org/3/library/functools.html). The following arguments were used in the **@nb.jit** decorator:

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
   5. Wikipedia contributors. (2024). *Wave function*. In Wikipedia, The Free Encyclopedia. Retrieved from https://en.wikipedia.org/wiki/Wave_function
   6. Zwiebach, B. (2022). *Mastering Quantum Mechanics: Essentials, Theory, and Applications*. MIT Press. ISBN: [026204613X](https://www.worldcat.org/isbn/026204613X)([9780262046138](https://www.worldcat.org/isbn/9780262046138))
   7. Bowers, P. L. (2020). *Lectures on Quantum Mechanics: A Primer for Mathematicians*. Cambridge University Press. ISBN: [1108429769](https://www.worldcat.org/isbn/1108429769) ([9781108429764](https://www.worldcat.org/isbn/9781108429764))
   8. Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 015402. doi:10.1088/1361-6404/aa9584
   9. Olver, F. W. J., & Maximon, L. C. (2010). NIST Handbook of Mathematical Functions. Cambridge University Press.
   10. CORDEIRO, MATHEUS; BEZERRA, I. P. ; VASCONCELOS, H. H. M. . Efficient Computation of the Wave Function ψn(x) using Hermite Coefficient Matrix in Python. In: 7º Workshop Escola de Computação e Informação Quântica (7ª WECIQ), 2024, Rio de Janeiro. ANAIS DO 7º WORKSHOP-ESCOLA DE COMPUTAÇÃO E INFORMAÇÃO QUÂNTICA. Rio de Janeiro: CEFET/RJ, 2024. p. 56-60. 

## 🤝 Contributing

Contributions, whether filing an issue, proposing improvements, or submitting a pull request, are welcome! Please feel free to explore, ask questions, and share your ideas.

## 📜 License

This project is available under the *BSD 3-Clause License*. See the LICENSE file for more details.

## 📬 Contact

If you have any questions or want to reach out to the team, please send me an email at [matheusgomescord@gmail.com](matheusgomescord@gmail.com).
