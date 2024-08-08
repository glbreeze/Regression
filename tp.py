import numpy as np
from sklearn.datasets import make_regression

#  MLP (FC, BN, ReLU, FC, BN, ReLU, FC  FC)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mean = (0, 0)
cov = [[1, 0.75],
       [0.75, 1]]
data = np.random.multivariate_normal(mean, cov,size=1000)
var = multivariate_normal(mean=mean, cov=cov)


x, y = np.mgrid[-1:1:.01, -1:1:.01]
rv = multivariate_normal([0, 0], [[1.0, 0], [0, 1.0]])
data = np.dstack((x, y))
z = rv.pdf(data)
plt.contourf(x, y, z, cmap='coolwarm')
plt.show()

dd = 100000000
num_param = 135168
for d in range(1000, 20000, 10):
       num_p = 11 * d + d*2
       dist = abs(num_param - num_p)
       if dist<= dd:
              dd = dist
              print('d {} num of param is {}'.format(d, num_p))

def matrix_with_angle(angle=np.pi/4, dim=256):
    vec1 = np.random.randn(dim)
    vec1 /= np.linalg.norm(vec1)  # Normalize to make it a unit vector

    random_vec = np.random.randn(dim)
    orthogonal_vec = random_vec - vec1 * np.dot(random_vec, vec1)
    orthogonal_vec /= np.linalg.norm(orthogonal_vec)  # Normalize to make it a unit vector

    vec2 = np.cos(angle) * vec1 + np.sin(angle) * orthogonal_vec
    return np.concatenate((vec1.reshape(1, -1), vec2.reshape(1, -1)), axis=0)