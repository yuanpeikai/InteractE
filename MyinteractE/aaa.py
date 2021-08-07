import torch
import numpy as np

all = []

list = np.array([1, 32, 53, 1, 5, 2, 43, 23])

a = np.where(list < 30, 1.0, 0.0).tolist()
b = np.where(list < 30, 1.0, 0.0).tolist()
print(a)
all.extend(a)
all.extend(b)
print(all)
