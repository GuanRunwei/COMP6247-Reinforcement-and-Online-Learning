import numpy as np
import matplotlib.pyplot as plt
# Set up synthetic data
#
N, p = 500, 30
X = np.random.randn(N, p)
wTrue = np.random.randn(p, 1)
yTarget = X @ wTrue + 0.8*np.random.randn(N,1)
# Initial guess and error
#
w0 = np.random.randn(p,1)
E0 = np.linalg.norm(yTarget - X @ w0)
# Parameters for gradient descent
#
MaxIter = 150
lRate = 0.001
Eplot = np.zeros((MaxIter, 1))
wIter = w0
for iter in range(MaxIter):
    wIter = wIter - lRate * X.T @ (X @ wIter - yTarget)
    Eplot[iter] = np.linalg.norm(X @ wIter - yTarget)
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(Eplot)
ax.set_xlabel("Iteration", fontsize=14)
ax.set_ylabel("Error", fontsize=14)
ax.grid(True)
ax.set_title("Gradient Descent on Linear Regression",
fontsize=16)
print("Residual Error (Initial) : %3.2f" %(E0))
print("Residual Error (Converged): %3.2f"
%(np.linalg.norm(X @ wIter - yTarget)))
plt.savefig("GDLR.png")
plt.show()
yPreds = X @ wIter

# Scatter plot of predictions against truth
#
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
plt.title("Linear Regression by GD")
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
plt.savefig("GD_prediction.png")
plt.show()
