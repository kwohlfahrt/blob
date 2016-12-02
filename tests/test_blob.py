from unittest import TestCase, skip
import numpy as np
import blob
from itertools import repeat

class TestLocalMinima(TestCase):
    def test_minima(self):
        data = np.ones((7, 7)) * 2
        data[2:5, 3:6] = 1
        data[3, 4] = 0
        # Diagonal values considered peaks
        expected = [[3, 4]]
        np.testing.assert_equal(blob.localMinima(data, 2), expected)

class TestDetection(TestCase):
    def test_blobs(self):
        image = np.zeros((128, 128), dtype='float')
        image[20, 50] = 1
        image[70:72, 10:12] = 1
        peaks = [[20, 50], [71, 11]]

        blobs = blob.findBlobs(image, scales=(1, 3), threshold=0.1)[:, 1:]
        np.testing.assert_equal(blobs, peaks)

if __name__ == "__main__":
    from unittest import main
    main()
