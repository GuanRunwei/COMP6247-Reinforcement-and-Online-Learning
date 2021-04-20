import numpy as np
import matplotlib.pyplot as plt




# number of data and dimension
#
N, p = 500, 30


# input data(covariates)
#
X = np.random.randn(N, p)

# True parameters
#
wTrue = np.random.randn(p, 1)

# set up targets(response)
#
yTarget = X @ wTrue + 0.8*np.random.randn(N, 1)

w0 = np.random.randn(p, 1)
E0 = np.linalg.norm(yTarget - X @ w0)

MaxIter = 2000
lRate = 0.01
Eplot = np.zeros((MaxIter, 1))

wIter = w0
print("Residual Error (initial):", E0)

for iter in range(MaxIter):
    j = np.floor(np.random.rand()*N).astype(int)

    xj = np.array([X[j, :]]).T
    yj = yTarget[j, :]
    yPred = xj.T @ wIter
    wIter = wIter - lRate * (yPred - yj) * xj
    Eplot[iter] = np.linalg.norm(yTarget - X @ wIter)

print("Residual Error (Converged):", np.linalg.norm(yTarget - X @ wIter))
yTarget = [i[0] for i in yTarget]
yPreds = X @ wIter
fig, ax = plt.subplots()
ax.plot(Eplot)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.grid(True)
ax.set_title("Stochastic Gradient Descent on Linear Regression")
plt.savefig("Stochastic Gradient Descent on Linear Regression_update.png")
plt.show()
# Scatter plot of predictions against truth
#
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
plt.title("Linear Regression by SGD")
ax[0].scatter(yTarget, yPreds, c='m', s=30)
ax[0].grid(True)
ax[0].set_xlabel("True Target", fontsize=14)
ax[0].set_ylabel("Prediction", fontsize=14)

ax[1].scatter(wTrue, wIter)
print(wTrue.shape)
print(wIter.shape)
ax[1].grid(True)
ax[1].set_xlabel("True Weights", fontsize=14)
ax[1].set_ylabel("Estimated Weights", fontsize=14)
plt.tight_layout()
plt.savefig("SGD_prediction_update.png")
plt.show()