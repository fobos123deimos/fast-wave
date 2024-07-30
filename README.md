


![Fast_Wave_logo](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/e1de91d2-3792-4b21-9553-7c13ce372a76)


![Version](https://img.shields.io/badge/version-1.1.0-blue.svg?cacheSeconds=2592000) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/blob/main/LICENSE)


<br>

> Harnessing the Power of the wavefunctions to navigate the quantum realm.🚀🌌

This package is an innovative project that delves into the complexities of quantum mechanics with its implementation for the time-independent wavefunction of the Quantum Harmonic Oscillator, a model widely used in Photonic Quantum Computing, making its calculations more efficient, and accurate! 🎉

## 📑 Table of Contents

- [Advantages](#-advantages)
- [Setup](#-setup)
- [Examples](#-exemples)
- [Theory](#-theory)
- [References](#-theory)
- [Contact](#-contact)

## ✨ Advantages


- **Highly Efficient**: The package uses numba's Just-in-Time (JIT) compilation in all its functions, increasing execution speed.
- **Highly Accurate**: As funções neste pacote têm precisão próxima à precisão do Wolfram Mathematica e do MATLAB.
- **Past response cache**: The functions in this package can use decorators of the Least Recently Used (LRU) type called lru_cache from the Python functools library to store previous results avoiding recalculation.


## 🛠 Setup
To use this package, simply run the following command in the command line: 
```bash
pip install fast-wave
``` 

## 🎨 Examples

```python
>>> from fast_wave.wavefunction import *
Functionality Test Passed: True
>>> hermite_sympy(2)
4*x**2 - 2
>>> create_hermite_coefficients_matrix(3)
array([[  0.,   0.,   0.,   1.],
       [  0.,   0.,   2.,   0.],
       [  0.,   4.,   0.,  -2.],
       [  8.,   0., -12.,   0.]])
>>> wave_smod = wavefunction(s_mode = True, o_dimensional = True, complex_bool = False, cache = False, cache_size = 128)
>>> wave_smmd = wavefunction(s_mode = True, o_dimensional = False, complex_bool = False, cache = False, cache_size = 128)
>>> wave_mmod = wavefunction(s_mode = False, o_dimensional = True, complex_bool = False, cache = False, cache_size = 128)
>>> wave_mmmd = wavefunction(s_mode = False, o_dimensional = False, complex_bool = False, cache = False, cache_size = 128)
>>> c_wave_smod = wavefunction(s_mode = True, o_dimensional = True, complex_bool = True, cache = False, cache_size = 128)
>>> c_wave_smmd = wavefunction(s_mode = True, o_dimensional = False, complex_bool = True, cache = False, cache_size = 128)
>>> c_wave_mmod = wavefunction(s_mode = False, o_dimensional = True, complex_bool = True, cache = False, cache_size = 128)
>>> c_wave_mmmd = wavefunction(s_mode = False, o_dimensional = False, complex_bool = True, cache = False, cache_size = 128)
>>> wave_smod(0, 1.0)
0.45558067201133257
>>> wave_smod(61, 1.0)
-0.2393049199171131
>>> c_wave_smod(0,1.0+2.0j)
(-1.4008797330262455-3.0609780602975003j)
>>> c_wave_smod(61,1.0+2.0j)
(-511062135.47555304+131445997.75753704j)
>>> wave_smmd(0,np.array([1.0,2.0]))
array([0.45558067, 0.10165379])
>>> wave_smmd(61,np.array([1.0,2.0]))
array([-0.23930492, -0.01677378])
>>> c_wave_smmd(0,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
array([ 0.40583486-0.63205035j, -0.49096842+0.56845369j])
>>> c_wave_smmd(61,np.array([1.0 + 1.0j, 2.0 + 2.0j]))
array([-7.56548941e+03+9.21498621e+02j, -1.64189542e+08-3.70892077e+08j])
>>> wave_mmod(1,1.0)
array([0.45558067, 0.64428837])
>>> c_wave_mmod(1,1.0 +2.0j)
array([-1.40087973-3.06097806j,  6.67661026-8.29116292j])
>>> wave_mmmd(1,np.array([1.0 ,2.0]))
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
>>> c_wave_mmmd(1,np.array([1.0 + 1.0j,2.0 + 2.0j]))
array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
       [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
```

## 📚 Theory

### Hermite Polynomials

The Hermite Polynomials, $H_n(x)$, are a sequence of orthogonal polynomials that arise in the solution of the Hermite differential equation:

$$
H_n''(x) - 2xH_n'(x) + 2nH_n(x) = 0
$$

where $n$ is a non-negative integer.

### Recursive Definition

The Hermite Polynomials can be defined recursively as follows:

$$
H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)
$$

with the initial conditions:

$$
H_0(x) = 1 \quad \text{and} \quad H_1(x) = 2x.
$$

### Rodrigues' Formula

An elegant method to generate Hermite Polynomials is through Rodrigues' formula:

$$
H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}(e^{-x^2})
$$

### Progression

Here are the first four Hermite Polynomials:

- $H_0(x) = 1$
- $H_1(x) = 2x$
- $H_2(x) = 4x^2 - 2$
- $H_3(x) = 8x^3 - 12x$

### Applications of the Hermite Polynomial

Hermite Polynomials play a crucial role in various areas of physics and mathematics, including quantum mechanics, where they are used in the wave functions of the quantum harmonic oscillator.

### $\star$ *Inside the Package*

The idea of ​​this package is to use a matrix with Hermite coefficients for sigle_mode problems up to $\mathbf{n\le 60}$ through two functions: 

- `wavefunction_smod(n,x)` $\mathbf{→}$ *[Single-Mode & Onedimensional]* 
- `wavefunction_smmd(n,xv)` $\mathbf{→}$ *[Single-Mode & Multidimensional]*

The use of this matrix of coefficients is only used up to a value of **60** (a value obtained empirically) because from this level onwards the function may present precision errors in its calculations with incoherent results. Here is an equation that represents this calculation:

- $C_{n}[i]•x^{p}_{i}$ $→$ *[Single-Mode & Onedimensional]*
- $C_{n}[i]•x^{p}_{ij}$ for each $x_j \in xv$ $→$ *[Single-Mode & Multidimensional]*

Where $\mathbf{x^{p}}$ is a vector of powers up to **n** and with zeros where there are no coefficients, for example $\mathbf{x^{p}}$ for the polynomial $\mathbf{H_{3}(x)}$ is equal to $\mathbf{x^{p} = [x^{3},0.0,x^{1},0.0]}$. On the other hand, $\mathbf{C_{n}[i]}$ is the row of coefficients for a degree $i$ of the Hermite polynomial for a matrix of Hermite coefficients going up to degree $n$. For this algorithm to perform as efficiently as possible, [Numba's Just-in-Time compilation](https://numba.pydata.org/) is used in conjunction with [lru_cache (Least Recently Used - Cache Management)](https://docs.python.org/3/library/functools.html). The arguments used in the **@jit** decorator were these:

- **nopython=True:** This argument forces the Numba compiler to operate in "nopython" mode, which means that all the code within the function must be compilable to pure machine code without falling back to the Python interpreter. This results in significant performance improvements by eliminating the overhead of the Python interpreter.
- **looplift=True:** This argument allows Numba to "lift" loops out of "nopython" mode. That is, if there are loops in the code that cannot be compiled in "nopython" mode, Numba will try to move them outside of the compiled part and execute them as normal Python code.
- **nogil=True:** This argument releases the Python Global Interpreter Lock (GIL) while the function is executing. It is useful for allowing the Numba-compiled code to run in parallel with other Python threads, increasing performance in multi-threaded programs.
- **boundscheck=False:** Disables array bounds checking. Normally, Numba checks if array indices are within valid bounds. Disabling this check can increase performance but may result in undefined behavior if there are out-of-bounds accesses.
- **cache=True:** Enables caching of the compiled function. The first time the function is compiled, Numba stores the compiled version in a cache. On subsequent executions, Numba can reuse the compiled version from the cache instead of recompiling the function, reducing the function's startup time.

The **@lru_cache(maxsize=128)** decorator in Python is used to apply the **Least Recently Used (LRU)** caching to a function. This caching mechanism can significantly improve the performance of functions that are called repeatedly with the same arguments by storing (caching) the results of expensive or frequently called functions and reusing the cached result when the same inputs occur again. The **maxsize** parameter helps manage memory usage by limiting the number of items stored in the cache. Once the cache reaches this limit, the least recently used items are discarded, keeping the cache size under control.

### Wavefunction

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

where $n$ is a non-negative integer, $m$ is the mass of the particle, $\omega$ is the angular frequency of the oscillator, and $H_n$ are the Hermite polynomials.

### Applications of the Wavefunction

Wavefunctions and the Schrödinger equation are central to understanding phenomena such as superposition, entanglement, and quantum tunneling, providing deep insights into the behavior of atoms, molecules, and subatomic particles.

### $\star$ *Inside the Package*

The idea of ​​this package is to use a recurrence to Wavefunction for sigle_mode problems where $\mathbf{n> 60}$, and for multi_mode problems to all values of $\mathbf{n}$ through these functions:

- `wavefunction_smod(n,x)` $\mathbf{→}$ *[Single-Mode & Onedimensional]* 
- `wavefunction_smmd(n,xv)` $\mathbf{→}$ *[Single-Mode & Multidimensional]*
- `wavefunction_mmod(n,x)` $\mathbf{→}$ *[Multi-Mode & Onedimensional]*
- `wavefunction_mmmd(n,xv)` $\mathbf{→}$ *[Multi-Mode & Multidimensional]*

Here's a way to get to recurrence:

<img src="https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/79140387-14e3-4250-ba46-918708bfc15b" alt="wavefunction_recurrence" width="1200">


Multi-Mode functions also use the Numba decorator with the same arguments, in addition to using the lru_cache decorator with **max_size = 128**.

### The Essence of the Package: *"Sigle-mode Problem."*

<br>

<img src="https://github.com/user-attachments/assets/56edb956-742b-4e17-af0e-7169207a455c" alt="Screenshot 2024-07-14 at 18-51-19 Captioned algorithm algpseudocode example - Online LaTeX Editor Overleaf" width="600">

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
