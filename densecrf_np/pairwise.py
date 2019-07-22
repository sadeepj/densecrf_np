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

from enum import Enum

import numpy as np

from densecrf_np.py_permutohedral import PyPermutohedral


class NormType(Enum):
    NO_NORMALIZATION = 0
    NORMALIZE_SYMMETRIC = 1
    NORMALIZE_BEFORE = 2
    NORMALIZE_AFTER = 2


class Pairwise(object):

    def apply(self, input_):
        pass


class SpatialPairwise(Pairwise):

    def __init__(self, image, sx, sy, norm_type=NormType.NORMALIZE_SYMMETRIC):
        h, w, _ = image.shape
        x = np.arange(w, dtype=np.float32)
        xx = np.tile(x, [h, 1]) / sx

        y = np.arange(h, dtype=np.float32).reshape((-1, 1))
        yy = np.tile(y, [1, w]) / sy

        self.features = np.stack([xx, yy], axis=2)
        self.lattice = PyPermutohedral()
        self.lattice.init(self.features, num_dimensions=2, num_points=h * w)

        all_ones = np.ones((h, w, 1), dtype=np.float32)
        self.norm = np.zeros((h, w, 1), dtype=np.float32)
        self.lattice.compute(self.norm, all_ones, 1, False)

        assert norm_type == NormType.NORMALIZE_SYMMETRIC

        self.norm = 1.0 / np.sqrt(self.norm + 1e-20)

    def apply(self, input_):
        input_ = input_ * self.norm
        output = np.zeros_like(input_, dtype=np.float32)
        self.lattice.compute(output, input_, input_.shape[-1], False)
        return output * self.norm


class BilateralPairwise(Pairwise):

    def __init__(self, image, sx, sy, sr, sg, sb, norm_type=NormType.NORMALIZE_SYMMETRIC):
        h, w, _ = image.shape
        x = np.arange(w, dtype=np.float32)
        xx = np.tile(x, [h, 1]) / sx  # TODO(sadeep) Possible optimisation as this was calculated in the Gaussian kernel

        y = np.arange(h, dtype=np.float32).reshape((-1, 1))
        yy = np.tile(y, [1, w]) / sy

        rgb = (image / [sr, sg, sb]).astype(np.float32)

        xy = np.stack([xx, yy], axis=2)
        self.features = np.concatenate([xy, rgb], axis=2)

        self.lattice = PyPermutohedral()
        self.lattice.init(self.features, num_dimensions=5, num_points=h * w)

        all_ones = np.ones((h, w, 1), dtype=np.float32)
        self.norm = np.zeros((h, w, 1), dtype=np.float32)
        self.lattice.compute(self.norm, all_ones, 1, False)

        assert norm_type == NormType.NORMALIZE_SYMMETRIC

        self.norm = 1.0 / np.sqrt(self.norm + 1e-20)

    def apply(self, input_):
        input_ = input_ * self.norm
        output = np.zeros_like(input_, dtype=np.float32)
        self.lattice.compute(output, input_, input_.shape[-1], False)
        return output * self.norm
