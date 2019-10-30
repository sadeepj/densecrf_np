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

from densecrf_np.pairwise import SpatialPairwise, BilateralPairwise
from densecrf_np.params import DenseCRFParams
from densecrf_np.util import softmax


class DenseCRF(object):

    def __init__(self, image, params: DenseCRFParams):
        alpha, beta, gamma = params.alpha, params.beta, params.gamma

        self.sp = SpatialPairwise(image, gamma, gamma)
        self.bp = BilateralPairwise(image, alpha, alpha, beta, beta, beta)

        self.spatial_weight = params.spatial_ker_weight
        self.bilateral_weight = params.bilateral_ker_weight

    def infer(self, unary_logits, num_iterations=5):
        q = softmax(unary_logits)

        for _ in range(num_iterations):
            tmp1 = unary_logits

            output = self.sp.apply(q)
            tmp1 = tmp1 + self.spatial_weight * output  # Do NOT use the += operator here!

            output = self.bp.apply(q)
            tmp1 = tmp1 + self.bilateral_weight * output

            q = softmax(tmp1)

        return q
