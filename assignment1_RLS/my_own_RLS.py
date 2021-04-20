import numpy as np
import matplotlib.pyplot as plt


def my_rls(X, yTarget):
    P = np.identity(X.shape[1]) * 2
    Lambda = 0.95
    w = np.random.randn(p, 1)
    Eplot = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        X_item = X[i]
        k = np.matrix((1/Lambda)*np.dot(X_item, P) / (1 + (1/Lambda) * X_item @ P @ X_item.T))
        e = (yTarget[i] - np.dot(X_item, w))
        w = w + k.T * e
        P = (1/Lambda) * P - (1/Lambda) * np.dot(P, np.dot(np.matrix(X_item).T, k))
        Eplot[i] = np.linalg.norm(yTarget - X @ w)
    return w, Eplot


if __name__ == '__main__':
    # number of data and dimension
    #
    N, p = 500, 120

    # input data(covariates)
    #
    X = np.random.randn(N, p)

    # True parameters
    #
    wTrue = np.random.randn(p, 1)

    # set up targets(response)
    #
    yTarget = X @ wTrue + 0.5*np.random.randn(N, 1)
    w, Eplot = my_rls(X, yTarget)
    yPredict = X @ w
    yTarget = np.asarray(yTarget.T[0])
    yPredict = np.asarray(yPredict.T[0])[0]
    w = np.asarray(w.T[0])
    wTrue = np.asarray(wTrue.T[0])
    print("yPredict", yPredict)
    fig, ax = plt.subplots()
    ax.plot(Eplot)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.grid(True)
    ax.set_title("My Recursive Least Squares on Linear Regression(120-dimension)")
    plt.savefig("120-dimension_data.png")
    plt.show()

    # Scatter plot of predictions against truth
    #
    fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].scatter(yTarget, yPredict, c='m', s=30)
    ax[0].grid(True)
    ax[0].set_xlabel("True Target", fontsize=14)
    ax[0].set_ylabel("Prediction", fontsize=14)

    ax[1].scatter(wTrue, w)
    ax[1].grid(True)
    ax[1].set_xlabel("True Weights", fontsize=14)
    ax[1].set_ylabel("Estimated Weights", fontsize=14)
    plt.tight_layout()
    plt.show()

    # show results
    plt.figure(figsize=(15, 9))
    plt.subplot(211)
    plt.title("Targets & Prediction")
    plt.xlabel("samples")
    plt.plot(yTarget, "b", label="true - target")

    plt.legend()
    plt.subplot(212)
    plt.xlabel("samples")
    plt.plot(yPredict, "g", label="pre - output")
    plt.legend()
    plt.tight_layout()
    plt.show()

