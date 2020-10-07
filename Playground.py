

# %%
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sympy import symbols, diff
from sklearn.linear_model import LinearRegression


def f(x):
    return x**2 + x + 1


data = np.linspace(start=-3, stop=3, num=100)


def df(x):
    return 2 * x + 1


# Set figure size
plt.figure(figsize=[10, 5])

# First subplot
plt.subplot(1, 2, 1)
plt.plot(data, f(data))
plt.xlim([-3, 3])
plt.ylim([0, 10])

# Second subplot
plt.subplot(1, 2, 2)
plt.plot(data, df(data))
plt.grid()
plt.xlim([-3, 3])
plt.ylim([-10, 10])

plt.show()

# %%

# Gradient descent

new_x = 3
previous_x = 0
multiplier = 0.1
descent_arr = []

for n in range(150):
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - multiplier * gradient
    descent_arr.append(new_x)


np_descent_arr = np.array(descent_arr)

plt.plot(data, f(data))
plt.scatter(descent_arr, f(descent_arr), color='red')


# %%

# 3D Graphs

# The function
def t(x, y):
    r = 3**(-x**2 - y**2)
    return 1 / (r + 1)


# The data
x_data = np.linspace(start=-2, stop=2, num=200)
y_data = np.linspace(start=-2, stop=2, num=200)
x_data, y_data = np.meshgrid(x_data, y_data)

# The figure
fig = plt.figure(figsize=[16, 12])

# The 3rd Axis
ax = fig.gca(projection="3d")

ax.plot_surface(x_data, y_data, t(x_data, y_data), cmap=cm.coolwarm)
# %%

# Sympy stuff

a, b = symbols("x, y")

dt_a = diff(t(a, b), a)
dt_b = diff(t(a, b), b)
print(dt_a)
print(dt_b)

z = t(a, b).evalf(subs={a: 1.8, b: 1.0})

print(z)
# %%
x = np.array([0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]).reshape((7, 1))
y = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape((7, 1))

linear_regression = LinearRegression()

linear_regression.fit(x, y)
plt.plot(x, linear_regression.predict(x))
plt.scatter(x, y)

print(linear_regression.intercept_[0])
print(linear_regression.coef_[0][0])

# %%
data.corr()

# %%
# mask = np.zeros_like(data.corr())
