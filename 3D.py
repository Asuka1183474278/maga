import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)

ax.plot(-2 * np.ones(100), -2 * np.ones(100), z, 'r--')
ax.plot(2 * np.ones(100), -2 * np.ones(100), z, 'r--')
ax.plot(-2 * np.ones(100), 2 * np.ones(100), z, 'r--')
ax.plot(2 * np.ones(100), 2 * np.ones(100), z, 'r--')
ax.plot(x, -2 * np.ones(100), -2 * np.ones(100), 'r--')
ax.plot(x, 2 * np.ones(100), 2 * np.ones(100), 'r--')
ax.plot(x, 2 * np.ones(100), -2 * np.ones(100), 'r--')
ax.plot(x, -2 * np.ones(100), 2 * np.ones(100), 'r--')
ax.plot(-2 * np.ones(100), y, -2 * np.ones(100), 'r--')
ax.plot(2 * np.ones(100), y, 2 * np.ones(100), 'r--')
ax.plot(2 * np.ones(100), y, -2 * np.ones(100), 'r--')
ax.plot(-2 * np.ones(100), y, 2 * np.ones(100), 'r--')
ax.legend()

plt.show()