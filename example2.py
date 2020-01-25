import numpy as np

a = np.array([[1, 275],
              [1, 441],
              [1, 494],
              [1, 593],
              [2, 679],
              [2, 533],
              [2, 686],
              [3, 559],
              [3, 219],
              [3, 455],
              [4, 605],
              [4, 468],
              [4, 692],
              [4, 613]])

b = np.split(a[:, 1], np.cumsum(np.unique(a[:, 0], return_counts=True)[1])[:-1])