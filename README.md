
<div style="margin-left: -20px;"><img src="https://github.com/matheus123deimos/CoEfficients-Matrix-Wavefunction/assets/20157453/10d35b23-bafe-4423-be8a-60f838771b3d" alt="" width="900" height="400"></div>


# Efficient Wavefunction⚡🌊

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/pikachu123deimos/CoEfficients-Matrix-Wavefunction/blob/efficient_wavefunction/LICENSE)


<br>


<br>

> Harnessing the power wavefunctions to navigate the quantum realm. 🚀🌌

This repository is the home to an innovative project that dives into the intricacies of quantum mechanics, utilizing wavefunctions to model and solve complex problems. Our mission is to make quantum calculations accessible, efficient, and fun! 🎉

## 📑 Table of Contents

- [Theory](#theory)
- [Advantages](#advantages)
- [Setup](#setup)
- [Examples](#exemples)
- [Contact](#contact)

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

### Applications

Hermite Polynomials play a crucial role in various areas of physics and mathematics, including quantum mechanics, where they are used in the wave functions of the quantum harmonic oscillator.

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

### Applications

Wavefunctions and the Schrödinger equation are central to understanding phenomena such as superposition, entanglement, and quantum tunneling, providing deep insights into the behavior of atoms, molecules, and subatomic particles.

## ✨ Advantages


- **Adjusting numba settings to optimize parallel computations**: Ensures that our mathematical operations are as efficient as possible.
- **Basic functionality test**: Demonstrates the core capabilities of our project with a simple yet effective test case.

## 🛠 Setup
To use this module, simply run a **git clone**, enter the repository folder through the terminal, and install the **requirements.txt**. After doing this, you can easily execute the module's functions using Python's IDE (**Python 3.11.5**) through the terminal in the repository folder or by placing scripts/notebooks that import the module's libraries inside the repository folder (⚠️ the repository folder, not the module one):
```bash
git clone https://github.com/matheus123deimos/CoEfficients-Matrix-Wavefunction.git
cd CoEfficients-Matrix-Wavefunction
pip install -r requirements.txt
``` 

## 🎨 Examples

```python
>>> from efficient_wavefunction.wavefunction import *
Functionality Test Passed: True
>>> hermite_sympy(2)
4*x**2 - 2
>>> create_hermite_coefficients_table(3)
array([[  0.,   0.,   0.,   1.],
       [  0.,   0.,   2.,   0.],
       [  0.,   4.,   0.,  -2.],
       [  8.,   0., -12.,   0.]])
 >>> wavefunction_smud(0, 1.0)
0.45558067201133257
>>> wavefunction_smod(61, 1.0)
-0.2393049199171131
>>> c_wavefunction_smod(0,1.0+2.0j)
(-1.4008797330262455-3.0609780602975003j)
>>> c_wavefunction_smod(61,1.0+2.0j)
(-511062135.47555304+131445997.75753704j)
>>> wavefunction_smmd(0,(1.0,2.0))
array([0.45558067, 0.10165379])
>>> wavefunction_smmd(61,(1.0,2.0))
array([-0.23930492, -0.01677378])
>>> c_wavefunction_smmd(0,(1.0 + 1.0j, 2.0 + 2.0j))
array([ 0.40583486-0.63205035j, -0.49096842+0.56845369j])
>>> c_wavefunction_smmd(61,(1.0 + 1.0j, 2.0 + 2.0j))
array([-7.56548941e+03+9.21498621e+02j, -1.64189542e+08-3.70892077e+08j])
>>> wavefunction_mmod(1,1.0)
array([0.45558067, 0.64428837])
>>> c_wavefunction_mmod(1,1.0 +2.0j)
array([-1.40087973-3.06097806j,  6.67661026-8.29116292j])
>>> wavefunction_mmmd(1,(1.0 ,2.0))
array([[0.45558067, 0.10165379],
       [0.64428837, 0.28752033]])
>>> c_wavefunction_mmmd(1,(1.0 + 1.0j,2.0 + 2.0j))
array([[ 0.40583486-0.63205035j, -0.49096842+0.56845369j],
       [ 1.46779135-0.31991701j, -2.99649822+0.21916143j]])
```


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

This project is available under the MIT License. See the LICENSE file for more details.

## 📬 Contact

If you have any questions or want to reach out to the team, please send me an email at .

---

Enjoy exploring the quantum world with us! 🌈✨
