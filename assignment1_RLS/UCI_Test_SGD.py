import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def my_sgd(X, yTarget):
    w0 = np.random.randn(X.shape[1], 1)
    E0 = np.linalg.norm(yTarget - X @ w0)
    print(E0)

    MaxIter = 5000
    lRate = 0.05
    Eplot = np.zeros((MaxIter, 1))

    wIter = w0
    for iter in range(MaxIter):
        j = np.floor(np.random.rand() * X.shape[0]).astype(int)
        xj = np.array([X[j, :]]).T
        yj = yTarget[j, :]
        yPred = xj.T @ wIter
        wIter = wIter - lRate * (yPred[0][0] - yj) * xj
        # print(wIter)
        Eplot[iter] = np.linalg.norm(yTarget - X @ wIter)
    return wIter, Eplot


hardware_data = pd.read_table("machine.data", encoding='UTF8', sep=',', header=None)

X_data = hardware_data.iloc[:, [2, 3, 4, 5, 6, 7]].values[16:]
X_data = normalize(X_data, axis=0, norm='max')

y_true = hardware_data.iloc[:, [-2]].values[16:]

y_predict = hardware_data.iloc[:, [-1]].values[16:]
w, Eplot = my_sgd(X_data, y_true)
y_true = np.asarray(y_true.T[0])
y_predict_sgd = X_data @ w
y_predict_rls = np.asarray(y_predict_sgd.T[0])
wEst = np.linalg.inv(X_data.T @ X_data) @ X_data.T @ y_true
w = np.asarray(w.T[0])
print(wEst)
print("print(Eplot[-1])", Eplot[-1])
fig, ax = plt.subplots()
ax.plot(Eplot)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.grid(True)
ax.set_title("Stochastic Gradient Descent on UCI data set")
plt.savefig("SGD_UCI.png")
plt.show()


# Scatter plot of predictions against truth
#
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter(y_true, y_predict_rls, c='m', s=30)
ax[0].grid(True)
ax[0].set_xlabel("True Target", fontsize=14)
ax[0].set_ylabel("Prediction", fontsize=14)
print(len(w))
print(wEst)
ax[1].scatter(w, wEst, c='m', s=30)
ax[1].grid(True)
ax[1].set_xlabel("True Weights", fontsize=14)
ax[1].set_ylabel("Estimated Weights", fontsize=14)
plt.tight_layout()
ax[0].set_title("Prediction by Stochastic Gradient Descent")
plt.savefig("SGD_UCI_prediction.png")
plt.show()