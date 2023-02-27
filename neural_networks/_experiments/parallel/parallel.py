from multiprocessing import Pool
import time


def f(x):
  return x * x


def task(xs):
  for x in xs:
    f(x)


if __name__ == "__main__":
  n = 100000000
  arguments = range(n)

  start = time.time()
  for argument in arguments:
    f(argument)
  print('Time taken:', time.time() - start)

  start = time.time()
  with Pool(20) as p:
    print('Time creating a pool:', (time.time() - start))

    # Divide the work between processes
    chunk_size = int(n/100)
    chunks = [arguments[x:x+chunk_size]
              for x in range(0, len(arguments), chunk_size)]
    start = time.time()
    p.map(task, chunks)
    print('Time taken:', time.time() - start)
