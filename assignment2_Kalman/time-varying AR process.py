import numpy as np
import matplotlib.pyplot as plt


# length of time series
N = 400

# gaussian random numbers as an excitation signal
ex = np.random.randn(N)

# second order AR process with coefficients slowly changing in time
a0 = np.array([1.2, -0.4])
A = np.zeros((N, 2))
omega, alpha = N/2, 0.1


for n in range(N):
    A[n, 0] = a0[0] + alpha * np.cos(2 * np.pi * n / N)
    A[n, 1] = a0[1] + alpha * np.sin(np.pi * n / N)


S = ex.copy()
for n in range(2, N):
    x = np.array([S[n-1], S[n-2]])
    S[n] = np.dot(x, A[n, :]) + ex[n]


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,4))
plt.tight_layout()
ax[1,0].plot(range(N), A[:,0])
ax[1,0].grid(True)
ax[1,0].set_title("Coefficient a0", color='m')
ax[1,1].plot(range(N), A[:,1], color='m')
ax[1,1].grid(True)
ax[1,1].set_title("Coefficient a1", color='m')
ax[0,0].plot(range(N), ex)
ax[0,0].grid(True)
ax[0,0].set_title("Random Excitation Signal")
ax[0,1].plot(range(N), S, color='m')
ax[0,1].grid(True)
ax[0,1].set_title("Time Varying Autoregressive Process")
plt.show()