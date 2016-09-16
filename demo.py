#!/usr/bin/env python3

from skimage.data import hubble_deep_field
from matplotlib import pyplot as plt
import blob

data = hubble_deep_field().astype('float32').sum(axis=-1)

# Equivalent to `blobs --size 3 20 --threshold 5 hubble_deep_field.tif`
blobs = blob.findBlobs(data, scales=range(5, 10), threshold=10)

fig, ax = plt.subplots(1, 1)
ax.imshow(data, cmap='gray')
ax.scatter(*blobs.T[1:][::-1], s=blobs.T[0] * 10, edgecolor='red', facecolor='none')
fig.tight_layout()
plt.show()
