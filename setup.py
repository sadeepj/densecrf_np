from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='densecrf_np',
    ext_modules=cythonize(
        Extension(
            "densecrf_np.py_permutohedral",
            sources=["densecrf_np/py_permutohedral.pyx"],
            include_dirs=[np.get_include()]
        ),
        # include_path=['densecrf_np']
    ),
    install_requires=["numpy"]
)
