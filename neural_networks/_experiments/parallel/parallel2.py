from multiprocessing import Pool
import time
import numpy as np


def f(X, Y):
  for row in X:
    np.sin(row)


if __name__ == "__main__":
  a = np.ones((10000, 10000))

  start = time.time()
  result = np.sin(a)
  print('Time taken:', time.time() - start)

  start = time.time()
  with Pool() as p:
    # apply np.sin to a row
    result = p.map(np.sin, a)
    print(type(result), type(result[0]), result[0][0])
    print('Time taken:', time.time() - start)
