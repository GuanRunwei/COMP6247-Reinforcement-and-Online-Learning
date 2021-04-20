import numpy as np
import matplotlib.pyplot as plt


# length of time series
N = 400

# Gaussian random numbers as an excitation signal
ex = np.random.randn(N)

# second order AR Process
a = np.array([1.2, -0.4])

S = ex.copy()
for n in range(2, N):
    x = np.array([S[n-1], S[n-2]])
    S[n] = np.dot(x, a) + ex[n]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
plt.tight_layout()

ax[0].plot(range(N), ex)
ax[0].grid(True)
ax[0].set_title("Random Excitation Signal")
ax[1].plot(range(N), S, color='m')
ax[1].grid(True)
ax[1].set_title("Autoregressive Process")
plt.show()