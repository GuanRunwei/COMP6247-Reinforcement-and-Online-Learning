import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def my_rls(X, yTarget):
    P = np.identity(X.shape[1]) * 2
    Lambda = 0.99
    w = np.random.randn(X_data.shape[1], 1)
    Eplot = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        X_item = X[i]
        k = np.matrix((1/Lambda)*np.dot(X_item, P) / (1 + (1/Lambda) * X_item @ P @ X_item.T))
        e = (yTarget[i] - np.dot(X_item, w))
        w = w + k.T * e
        P = (1/Lambda) * P - (1/Lambda) * np.dot(P, np.dot(np.matrix(X_item).T, k))

        Eplot[i] = np.linalg.norm(yTarget - X @ w)
    return w, Eplot


hardware_data = pd.read_table("machine.data", encoding='UTF8', sep=',', header=None)

X_data = hardware_data.iloc[:, [2, 3, 4, 5, 6, 7]].values[16:]
y_true = hardware_data.iloc[:, [-2]].values[16:]
y_predict = hardware_data.iloc[:, [-1]].values[16:]
w, Eplot = my_rls(X_data, y_true)
wEst = np.linalg.inv(X_data.T @ X_data) @ X_data.T @ y_true

y_predict_rls = X_data @ w
y_true = np.asarray(y_true.T[0])
y_predict = np.asarray(y_predict.T[0])
y_predict_rls = np.asarray(y_predict_rls.T[0])
w = np.asarray(w.T[0])
wEst = np.asarray(wEst.T[0])
print(y_true)
print("print(Eplot[-1])", Eplot[-1])
fig, ax = plt.subplots()
ax.plot(Eplot)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.grid(True)
ax.set_title("Recursive Least Squares on UCI data set")
plt.savefig("RLS_UCI.png")
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
ax[1].set_xlabel("True Weights", fontsize=14 )
ax[1].set_ylabel("Estimated Weights", fontsize=14)
plt.tight_layout()
ax[0].set_title("Prediction by Recursive Least Squares")
plt.savefig("RLS_UCI_prediction.png")
plt.show()


# show results
plt.figure(figsize=(15, 9))
plt.subplot(211)
plt.title("Targets & Prediction")
plt.xlabel("samples")
plt.plot(y_true, "b", label="true - target")

plt.legend()
plt.subplot(212)
plt.xlabel("samples")
plt.plot(y_predict_rls[0], "g", label="pre - output")
plt.legend()
plt.savefig("Y-line_compare_myrls.png")
plt.show()








