# distutils: language = c++

"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool

from densecrf_np.Permutohedral cimport Permutohedral

cdef class PyPermutohedral:
    cdef Permutohedral c_obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init(self, np.ndarray[float, ndim=3, mode="c"] features not None, int num_dimensions, int num_points):
        self.c_obj.init(&features[0, 0, 0], num_dimensions, num_points)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute(self,
                np.ndarray[float, ndim=3, mode="c"] output not None,
                np.ndarray[float, ndim=3, mode="c"] inp not None,
                int value_size, bool reverse):
        self.c_obj.compute(&output[0, 0, 0], &inp[0, 0, 0], value_size, reverse)
