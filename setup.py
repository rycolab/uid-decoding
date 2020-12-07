from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
    ext_modules = cythonize(Extension("datastructures.sum_heap", ["datastructures/sum_heap.pyx"]))
)
