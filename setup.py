import setuptools
from setuptools import find_packages, Extension
from Cython.Build import cythonize
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

name = "fast_wave"
version = "1.6.9"
description = "Package for the calculation of the time-independent wavefunction." 
author_email = "matheusgomescord@gmail.com"
url = "https://github.com/pikachu123deimos/fast-wave" 

install_requires = [
    "llvmlite==0.42.0",
    "mpmath==1.3.0",
    "numba==0.59.1",
    "numpy==1.26.4",
    "sympy==1.12",
    "cython==3.0.10",
]

test_requires = [
    "pytest= ^7.1.2",
]

packages = find_packages(where='src')

package_data = {'fast_wave': ['*.pyd', '*.so', '*.pyx']}
extensions = [
    Extension("fast_wave.wavefunction_cython", ["src/fast_wave/wavefunction_cython.pyx"],include_dirs=[numpy.get_include()])
]

classifiers = [
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: BSD License',
    'Development Status :: 5 - Production/Stable',
]

setuptools.setup(
    name=name,
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    description=description,
    author=name,
    author_email=author_email,
    url=url,
    install_requires=install_requires,
    test_requires=test_requires,
    packages=packages,
    package_dir={'': 'src'},
    package_data=package_data,
    classifiers=classifiers,
)
