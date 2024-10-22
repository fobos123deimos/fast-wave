


![Fast_Wave_logo](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/e1de91d2-3792-4b21-9553-7c13ce372a76)


![Version](https://img.shields.io/badge/version-1.3.0-blue.svg?cacheSeconds=2592000) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/blob/main/LICENSE)


<br>

> Harnessing the Power of the wavefunctions to navigate the quantum realm.🚀🌌

This package represents the time-independent wavefunctions of the Quantum Harmonic Oscillator as Fock states, optimizing the accuracy and efficiency of calculations in Photonic Quantum Computing. 

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
- **Highly Accurate**: The functions in this package have precision next to the precision of Wolfram Mathematica and MATLAB. In addition, there is a module just for calculating wave functions with arbitrary-precision using the *mpmath* package.
- **Past response cache**: This package provides a caching module designed to enhance the performance of functions that take multiple positions of a *NumPy* array as input. By leveraging Python's functools.lru_cache, this module stores previously computed results, eliminating the need for redundant calculations. This caching module is inspired by the [caching module](https://github.com/XanaduAI/MrMustard/blob/develop/mrmustard/math/caching.py#L26) from Mr. Mustard, a package from the photonic quantum computing company Xanadu.


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
>>> wn.psi_n_multiple_fock_multiple_position_complex(1,np.array([1.0 + 1.0j,2.0 + 2.0j]))
array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
       [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
>>> wc.psi_n_multiple_fock_multiple_position(1,np.array([1.0 ,2.0]))
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
>>> wc.psi_n_multiple_fock_multiple_position_complex(1,np.array([1.0 + 1.0j,2.0 + 2.0j]))
array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
       [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
```

There are other examples in the examples folder: [Speed Tests: Numba & Cython](https://colab.research.google.com/github/fobos123deimos/fast-wave/blob/main/examples/speed_tests_numba_and_cython.ipynb); [Precision Tests: mpmath](https://colab.research.google.com/github/fobos123deimos/fast-wave/blob/main/examples/precision_tests_mpmath.ipynb). In the first one there is a comparison with the [Mr Mustard](https://mrmustard.readthedocs.io/en/stable/) package.

## 🌊 The Wavefunction

The wavefunction, $\psi(x)$, is a fundamental concept in quantum mechanics that describes the quantum state of a particle or system. The absolute square of the wavefunction, $|\psi(x)|^2$, represents the probability density of finding the particle at a position $x$.

### Schrödinger Equation

The behavior of a wavefunction is governed by the Schrödinger equation, a fundamental equation in quantum mechanics:

$$
i\hbar\frac{\partial}{\partial t}\psi(x,t) = \hat{H}\psi(x,t)
$$

where $i$ is the imaginary unit, $\hbar$ is the reduced Planck's constant, $\hat{H}$ is the Hamiltonian operator, and $\psi(x,t)$ is the wavefunction of the system at position $x$ and time $t$.

### Normalization

For the wavefunction to be physically meaningful, it must be normalized:

$$
\int_{-\infty}^{\infty} |\psi(x)|^2 dx = 1
$$

This ensures that the total probability of finding the particle somewhere in space is one.

### Quantum Harmonic Oscillator

The wavefunctions of the quantum harmonic oscillator, a system that models particles in a potential well, are given by:

$$
\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n\left(\sqrt{\frac{m\omega}{\hbar}}x\right) e^{-\frac{m\omega x^2}{2\hbar}}
$$

where $n$ is a non-negative integer, $m$ is the mass of the particle, $\omega$ is the angular frequency of the oscillator, and $H_n$ are the Hermite polynomials. We can use this wavefunction to describe Fock states. 

## 🔁 The Wavefunction Recurrence

Most algorithms in this package use a recurrence for the wave function. Here's a way to get to recurrence:

<img src="https://github.com/user-attachments/assets/e0d8fdcf-ddde-45b0-b56b-e70f0412680a" alt="wavefunction_recurrence" width="3000">



## ⚡️The Numba Module - Hybrid Solution

We use a hybrid solution with two forms for calculating the wave function for problems of Single Fock and Multiple Position (`psi_n_single_fock_multiple_position`). To $n>60$ or $x\\_size>35$, we use the recurrence for the wave function. To $n\le 60$ and $0<x\\_size\le35$ we use a precomputed matrix with the normalized coefficients of the Hermite polynomial as follows:


<img src="https://github.com/user-attachments/assets/0d248db6-acdc-42d6-a9c2-b1d600be8fee" alt="wavefunction_recurrence" width="500">


In this equation, $\mathbf{C^{s}_{n}[i]}$ represents the row of normalized coefficients for degree $i$ of the Hermite polynomial, within a matrix of Hermite normalized coefficients that extends up to degree $n$. On the other hand, $\mathbf{x^{p}}$ is a vector of powers up to n, with zeros in place of missing coefficients; for example, $\mathbf{x^{p}}$ is equal to $\mathbf{x^{p} = [x^{3}, 0.0, x^{1}, 0.0]}$. This hybrid algorithm is also used in Single Fock and Single Position (`psi_n_single_fock_single_position`) problems, though it offers no computational advantage in these cases. Additionally, there is an argument named **CS_matrix** for these Single Fock functions, set to **True** to enable the use of this matrix. In other words, you can use only the recurrence relation for the wave function at any value. The use of this coefficient matrix is limited to values up to **60** (determined empirically), as beyond this point, the function may encounter precision errors, resulting in incoherent outputs.

## ⚡️ The Numba Module - Arguments

For this algorithm to perform as efficiently as possible, [Numba's Just-in-Time compilation](https://numba.pydata.org/) is used in conjunction with [lru_cache (Least Recently Used - Cache Management)](https://docs.python.org/3/library/functools.html). The arguments used in the **@nb.jit** decorator were these:

- **nopython=True:** This argument forces the Numba compiler to operate in "nopython" mode, which means that all the code within the function must be compilable to pure machine code without falling back to the Python interpreter. This results in significant performance improvements by eliminating the overhead of the Python interpreter.
- **looplift=True:** This argument allows Numba to "lift" loops out of "nopython" mode. That is, if there are loops in the code that cannot be compiled in "nopython" mode, Numba will try to move them outside of the compiled part and execute them as normal Python code.
- **nogil=True:** This argument releases the Python Global Interpreter Lock (GIL) while the function is executing. It is useful for allowing the Numba-compiled code to run in parallel with other Python threads, increasing performance in multi-threaded programs.
- **boundscheck=False:** Disables array bounds checking. Normally, Numba checks if array indices are within valid bounds. Disabling this check can increase performance but may result in undefined behavior if there are out-of-bounds accesses.
- **cache=True:** Enables caching of the compiled function. The first time the function is compiled, Numba stores the compiled version in a cache. On subsequent executions, Numba can reuse the compiled version from the cache instead of recompiling the function, reducing the function's startup time.

## ⚙️ The Cython Module

The [Cython](https://cython.org/) module includes compiled files for Linux (**.so**) and Windows (**.pyd**), which allows it to be used in Google Colab (Linux). Additionally, this module supports three versions of Python 3: 3.10, 3.11, and 3.12. All these files are placed in the package folder upon installation. The source code of the Cython module is available in the repository in **.pyx** format. In the functions of the Cython module, some decorators are used to increase speed:

- **@cython.nogil**: This decorator allows a Cython function to release the Global Interpreter Lock (GIL), making it possible to execute that block of code concurrently in multiple threads.
- **@cython.cfunc**:  This decorator tells Cython to treat the function as a C function, meaning it can be called from other Cython or C code, not just Python. The function will have C-level calling conventions.
- **@cython.locals(...)**: Declares local variable types to optimize performance.
- **@cython.boundscheck(False)**: Disables bounds checking for arrays/lists to boost speed, but at the cost of safety. 


## 📖 References

Our journey through the quantum realm is inspired by the following seminal works:

- Wikipedia contributors. (2021). Hermite polynomials. In Wikipedia, The Free Encyclopedia. Retrieved from https://en.wikipedia.org/wiki/Hermite_polynomials
- Olver, F. W. J., & Maximon, L. C. (2010). NIST Handbook of Mathematical Functions. Cambridge University Press.
- NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.0.28 of 2020-09-15.
- Sympy Documentation: https://docs.sympy.org/latest/modules/polys/index.html
- Scipy Documentation on `eval_hermite`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html
- Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum harmonic oscillator. *European Journal of Physics*, 39(1), 015402. doi:10.1088/1361-6404/aa9584

## 🤝 Contributing

Contributions, whether filing an issue, proposing improvements, or submitting a pull request, are welcome! Please feel free to explore, ask questions, and share your ideas.

## 📜 License

This project is available under the *BSD 3-Clause License*. See the LICENSE file for more details.

## 📬 Contact

If you have any questions or want to reach out to the team, please send me an email at [matheusgomescord@gmail.com](matheusgomescord@gmail.com)
.

---

Enjoy exploring the quantum world! 🌈✨
