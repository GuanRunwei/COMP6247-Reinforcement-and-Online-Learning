import numpy as np
import matplotlib.pyplot as plt

# time series data y
# th_n_n: estimate at time n using all data upto time n
# th_n_n1: estimate at time n using all data upto time n-1


# initialize
#
x = np.zeros((2, 1))
th_n1_n1 = np.random.randn(2, 1)
P_n1_n1 = 0.001 * np.eye(2)

# Length of time series
#
N = 400

y = np.random.randn(N)
ePlot = np.zeros(N)


# Gaussian random numbers as an excitation signal
#
ex = np.random.randn(N)



# noise variances -- hyperparameters(to be tuned)
# set measurement noise as fraction of data variance (first few samples)
# guess for process noise
R = 0.2 * np.std(ex[0: 10])
beta = 0.0001
Q = beta * np.eye(2)

# space to store and plot
#
th_conv = np.zeros([2, N])

# First two estimates are initial guesses
#
th_conv[0, 0] = th_n1_n1[0]
th_conv[0, 1] = th_n1_n1[1]
th_conv[1, 0] = th_n1_n1[0]
th_conv[1, 1] = th_n1_n1[1]


# Kalman Iteration Loop (univariate observation, start from time step 2)
#
for n in range(2, N):
    x[0] = y[n - 1]  # input vector contains past values
    x[1] = y[n - 2]

    # prediction of state and covariance
    th_n_n1 = th_n1_n1.copy()
    P_n_n1 = P_n1_n1 + Q

    yh = th_n_n1.T @ x
    en = y[n] - yh
    ePlot[n] = en

    # Kalman gain(kn) and innovation variance(den)
    #
    den = x.T @ P_n1_n1 @ x + R
    kn = P_n1_n1 @ x /den

    # Posterior update
    th_n_n = th_n_n1 + kn * en
    P_n_n = (np.eye(2) - kn @ x.T) @ P_n_n1

    # save
    th_conv[0, n] = th_n_n[0]
    th_conv[1, n] = th_n_n[1]

    # Remember for next step
    #
    th_n1_n1 = th_n_n.copy()
    P_n1_n1 = P_n_n.copy()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].plot(th_conv[0])
ax[0].set_xlim(0, 500)
ax[0].axhline(y=a[0], color='r')
ax[1].plot(th_conv[1])
ax[1].set_xlim(0, 500)
ax[1].axhline(y=a[1], color='r')
ax[1].set_title("R = %4.3f, Q = %6.5f I" %(R, beta))
plt.show()


