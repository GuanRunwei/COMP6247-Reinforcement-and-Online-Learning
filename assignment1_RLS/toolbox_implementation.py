import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import padasip as pa
import cmath


hardware_data = pd.read_table("machine.data", encoding='UTF8', sep=',', header=None)

X_data = hardware_data.iloc[:, [2, 3, 4, 5, 6, 7]].values[16:]
y_true = hardware_data.iloc[:, [-2]].values[16:]
y_predict = hardware_data.iloc[:, [-1]].values[16:]
maxIter = 2000

fil = pa.filters.FilterRLS(n=len(X_data[0]), mu=1, w='random')
y_true = np.asarray(y_true.T[0])
X_data = np.asarray(X_data)
print(len(X_data))
yPredict, e, wPredict = fil.run(y_true, X_data)
Eplots = []
print(X_data[0])
print(wPredict.shape)
print(y_true.shape)
for i in range(len(X_data)):
    E_temp = np.linalg.norm(X_data @ wPredict[i].T - y_true)
    Eplots.append(E_temp)



fig, ax = plt.subplots()
ax.plot(Eplots)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
ax.grid(True)
ax.set_title("Recursive Least Squares(padasip) on UCI data set")
plt.savefig("Recursive Least Squares(padasip) on UCI data set.png")
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
plt.plot(yPredict, "g", label="pre - output")
plt.legend()
plt.tight_layout()
plt.savefig("Y-line_compare.png")
plt.show()
