from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='densecrf_np',
    version="0.0.1",
    author="Sadeep Jayasumana",
    author_email="sadeep@apache.org",
    ext_modules=cythonize(
        Extension(
            "densecrf_np.py_permutohedral",
            sources=["densecrf_np/py_permutohedral.pyx"],
            include_dirs=[np.get_include()]
        ),
    ),
)
