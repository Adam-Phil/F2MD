from sklearn.metrics import *
import numpy as np
import random

random.seed(10)


def flip(x, n):
    ret = []
    for i in range(len(x)):
        if i < n:
            if x[i] == 0:
                ret.append(1)
            else:
                ret.append(0)
        else:
            ret.append(x[i])
    return ret


if __name__ == "__main__":
    length = 35000000
    try_out = 500000
    prob = try_out / length
    one = np.array([0 if random.uniform(0, 1) >= prob else 1 for _ in range(length)])
    two = np.array([0 for _ in range(length)])
    print(log_loss(one, two))
    print(mean_squared_error(one, two))
