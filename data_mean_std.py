import torch
import numpy as np
from torchvision import datasets, transforms

POOL_DATASET_PATH = "data"

data = datasets.ImageFolder(root=POOL_DATASET_PATH, transform=transforms.ToTensor())

means = []
stds = []
for image, target in data:
    numpy_image = image.numpy()
    means.append(np.mean(numpy_image, axis=(1, 2)))
    stds.append(np.std(numpy_image, axis=(1, 2), ddof=1))
print('mean =', np.array(means).mean(axis=0))
print('std =', np.array(stds).mean(axis=0))
