# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

rng = default_rng()
# %%
x = np.linspace(start=0, stop=10, num=50)
y = np.linspace(start=0, stop=10, num=50)

for i in range(len(y)):
    noise = np.random.randint(low=-5, high=5)
    y[i] += noise

plt.scatter(x, y)
# %%

# Straight line function


def lin_reg(x, theta0, theta1):
    return theta0 + theta1 * x


plt.scatter(x, y)

# parameters
theta0 = 0
theta1 = 0

# Hyper parameters
m = len(y)
a = 0.01

for i in range(1000):
    # gradient = (lin_reg(x, theta0, theta1) - y).transpose().dot(x) / len(x)
    # theta0 -= a * gradient
    # theta1 -= a * gradient

    theta0 -= a * sum(lin_reg(x, theta0, theta1) - y) / m
    theta1 -= a * sum((lin_reg(x, theta0, theta1) - y) * x) / m

    # print(f"{theta0}, {theta1}")

plt.plot(x, lin_reg(x, theta0, theta1))
# %%
