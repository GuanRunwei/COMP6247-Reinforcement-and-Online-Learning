import numpy as np
import matplotlib.pyplot as plt

# number of data and dimension
N, p = 500, 30

# input data (covariances)
X = np.random.randn(N, p)

# true parameters
wTrue = np.random.randn(p, 1)

# set up targets(response)
yTarget = X @ wTrue + 0.8*np.random.randn(N, 1)
print(0.8*np.random.randn(N, 1))

# estimate the weights by pesudo inverse
wEst = np.linalg.inv(X.T @ X) @ X.T @ yTarget

# Predict from the model
yEst = X @ wEst

# Scatter plot of predictions against truth
#
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter(yTarget, yEst, c='m', s=30)
ax[0].grid(True)
ax[0].set_xlabel("True Target", fontsize=14)
ax[0].set_ylabel("Prediction", fontsize=14)

ax[1].scatter(wTrue, wEst)
ax[1].grid(True)
ax[1].set_xlabel("True Weights", fontsize=14)
ax[1].set_ylabel("Estimated Weights", fontsize=14)
plt.tight_layout()
plt.title("Linear Regression by Pesudo Inverse")
plt.savefig("LR_Pesudo Inverse.png")
plt.show()


# Error from the model
#
print("Residual Error: %3.2f" %(np.linalg.norm(yEst - yTarget)))