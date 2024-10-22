


![Fast_Wave_logo](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/e1de91d2-3792-4b21-9553-7c13ce372a76)


![Version](https://img.shields.io/badge/version-1.3.0-blue.svg?cacheSeconds=2592000) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/blob/main/LICENSE)


<br>

> Harnessing the Power of the wavefunctions to navigate the quantum realm.🚀🌌

This package represents the time-independent wavefunctions of the Quantum Harmonic Oscillator as Fock states, optimizing the accuracy and efficiency of calculations in Photonic Quantum Computing. 

## 📑 Table of Contents

- [Advantages](#-advantages)
- [Setup](#-setup)
- [Examples](#-exemples)
- [Theory](#-theory)
- [References](#-theory)
- [Contact](#-contact)

## ✨ Advantages


- **Highly Efficient**: This package includes two fixed-point modules focused on speed. One is implemented using *Numba*, an open-source Just-in-Time (JIT) compiler, and the other module is implemented in *Cython*, a programming language that combines the ease of use of Python with the speed of C..
- **Highly Accurate**: The functions in this package have precision next to the precision of Wolfram Mathematica and MATLAB. In addition, there is a module just for calculating wave functions with arbitrary-precision using the *mpmath* package.
- **Past response cache**: This package provides a caching module designed to enhance the performance of functions that take multiple positions of a *NumPy* array as input. By leveraging Python's functools.lru_cache, this module stores previously computed results, eliminating the need for redundant calculations.


## 🛠 Setup
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

## 📚 The Wavefunction

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

### The Wavefunction Recurrence

Most algorithms in this package use a recurrence for the wave function. Here's a way to get to recurrence:

<img src="https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/79140387-14e3-4250-ba46-918708bfc15b" alt="wavefunction_recurrence" width="1200">


### $\star$ *Inside the Package*

The idea of ​​this package is to use a matrix with Hermite coefficients for sigle_mode problems up to $\mathbf{n\le 60}$ through two functions: 

- `wavefunction_smod(n,x)` $\mathbf{→}$ *[Single-Mode & Onedimensional]* 
- `wavefunction_smmd(n,xv)` $\mathbf{→}$ *[Single-Mode & Multidimensional]*

The use of this coefficient matrix is ​​only used up to the value **60** (value obtained empirically) because from this level onwards the function may present precision errors in its calculations with incoherent results. Even so, there is a small imprecision around the **60th** degree for the coefficient matrix, which is why the functions that work with it have an argument named *more_fast* set to **True**, that is, it is faster but inaccurate around the **60th** degree. When **False**, the algorithm is a little slower but with high precision. Here is an equation that represents this calculation:

- $C_{n}[i]•x^{p}_{i}$ $→$ *[Single-Mode & Onedimensional]*
- $C_{n}[i]•x^{p}_{ij}$ for each $x_j \in xv$ $→$ *[Single-Mode & Multidimensional]*

Where $\mathbf{x^{p}}$ is a vector of powers up to **n** and with zeros where there are no coefficients, for example $\mathbf{x^{p}}$ for the polynomial $\mathbf{H_{3}(x)}$ is equal to $\mathbf{x^{p} = [x^{3},0.0,x^{1},0.0]}$. On the other hand, $\mathbf{C_{n}[i]}$ is the row of coefficients for a degree $i$ of the Hermite polynomial for a matrix of Hermite coefficients going up to degree $n$. For this algorithm to perform as efficiently as possible, [Numba's Just-in-Time compilation](https://numba.pydata.org/) is used in conjunction with [lru_cache (Least Recently Used - Cache Management)](https://docs.python.org/3/library/functools.html). The arguments used in the **@jit** decorator were these:

- **nopython=True:** This argument forces the Numba compiler to operate in "nopython" mode, which means that all the code within the function must be compilable to pure machine code without falling back to the Python interpreter. This results in significant performance improvements by eliminating the overhead of the Python interpreter.
- **looplift=True:** This argument allows Numba to "lift" loops out of "nopython" mode. That is, if there are loops in the code that cannot be compiled in "nopython" mode, Numba will try to move them outside of the compiled part and execute them as normal Python code.
- **nogil=True:** This argument releases the Python Global Interpreter Lock (GIL) while the function is executing. It is useful for allowing the Numba-compiled code to run in parallel with other Python threads, increasing performance in multi-threaded programs.
- **boundscheck=False:** Disables array bounds checking. Normally, Numba checks if array indices are within valid bounds. Disabling this check can increase performance but may result in undefined behavior if there are out-of-bounds accesses.
- **cache=True:** Enables caching of the compiled function. The first time the function is compiled, Numba stores the compiled version in a cache. On subsequent executions, Numba can reuse the compiled version from the cache instead of recompiling the function, reducing the function's startup time.


### $\star$ *Inside the Package*

The idea of ​​this package is to use a recurrence to Wavefunction for sigle_mode problems where $\mathbf{n> 60}$, and for multi_mode problems to all values of $\mathbf{n}$ through these functions:

- `wavefunction_smod(n,x)` $\mathbf{→}$ *[Single-Mode & Onedimensional]* 
- `wavefunction_smmd(n,xv)` $\mathbf{→}$ *[Single-Mode & Multidimensional]*
- `wavefunction_mmod(n,x)` $\mathbf{→}$ *[Multi-Mode & Onedimensional]*
- `wavefunction_mmmd(n,xv)` $\mathbf{→}$ *[Multi-Mode & Multidimensional]*


### The Essence of the Package: *"Sigle-mode Problem."*

<br>

<img src="https://github.com/user-attachments/assets/0d1cdeb2-1912-4794-b321-832ec2c0b3fd" alt="Fast Wave to the Single-Mode Problems" width="600">

<br>
<br>

This algorithm is a representation of the use of all functions of the package to solve single-mode problems. To solve multi-mode problems we use only two functions that are based on the strategies to $n>60$. We can estimate the time complexity for $n\leq60$ with $O(1)$ for the onedimensional case and $O(\mathbf{x}.size)$ for the multidimensional case, just as we can estimate the space complexity with $O(1)$ for the onedimensional case and $O(\mathbf{x}.size)$ for the multidimensional case (result vector allocation). For the case of $n>60$, we have a time complexity value equal to $O(n)$ for the onedimensional case and $O(n*\mathbf{x}.size)$ for the multidimensional case, as well as we have a space complexity of $O(n)$ for the onedimensional case and $O(n*\mathbf{x}.size)$ for the multidimensional case (allocating the result matrix).

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
