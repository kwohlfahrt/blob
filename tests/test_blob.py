from unittest import TestCase, skip
import numpy as np
import blob
from itertools import repeat

class TestLaplacian(TestCase):
    def test_scale_invariance_1d(self):
        scale = 10.0

        for ndim in (1, 2, 3):
            for scale in (2, 5, 10, 20, 40):
                signal = np.zeros((200,) * ndim, dtype='float')
                peak_region = tuple(slice(s // 2 - scale, s // 2 + scale) for s in signal.shape)
                signal[peak_region] = 1
                psf = blob.gaussianPsfHat(signal.shape, repeat(scale), 'float')
                blurred = blob.ifftn(blob.fftn(signal) * psf).real
                print(ndim, scale, (blob.laplacianOperator(blurred) * scale ** 2).max())

class TestLocalMinima(TestCase):
    def test_minima(self):
        data = np.ones((7, 7)) * 2
        data[2:5, 3:6] = 1
        data[3, 4] = 0
        # Diagonal values considered peaks
        expected = [[2, 3], [2, 5], [3, 4], [4, 3], [4, 5]]
        np.testing.assert_equal(blob.localMinima(data, 2), expected)

@skip
class TestGradient(TestCase):
    def test_forward(self):
        data = np.array([1, 2, 3, 5, 7, 4, 4])
        expected = [1, 1, 2, 2, -3, 0, 0]
        np.testing.assert_equal(blob.gradient(data, 0), expected)

    def test_backward(self):
        data = np.array([1, 2, 3, 5, 7, 4, 4])
        expected = [1, 1, 1, 2, 2, -3, 0]
        np.testing.assert_equal(blob.gradient(data, 0, True), expected)

if __name__ == "__main__":
    from unittest import main
    main()
