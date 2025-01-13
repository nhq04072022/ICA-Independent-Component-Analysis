import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
# from sklearn.decomposition import PCA
from matplotlib import style

np.random.seed(0)


def plot_signal(W, X):
    estimate_S = W.T @ X
    estimate_S = estimate_S.T
    corlor = ["#fc4f30", "#008fd5", "#e5ae38"]
    plt.figure()
    for i in range(W.shape[1]):
        fig = plt.subplot(W.shape[1], 1, i + 1)
        fig.set_title(f"Independent component $s_{i+1}$")
        plt.plot(estimate_S[:, i], color=corlor[i])
        plt.tight_layout(pad=1)

    plt.show()


n_samples = 400
time = np.linspace(0, 10, n_samples)

s1 = 2.2 * np.sin(2 * time)
s2 = 0.8 * np.sign(np.sin(3 * time))
s3 = 1.5 * signal.sawtooth(2 * np.pi * time)

noise1 = 1.5 * np.random.normal(loc=0.0, scale=0.04, size=n_samples)
noise2 = 0.7 * np.random.normal(loc=0.0, scale=0.04, size=n_samples)
noise3 = 2.1 * np.random.normal(loc=0.0, scale=0.04, size=n_samples)

S = np.c_[s1 + noise1, s2 + noise2, s3 + noise3]

plt.figure()

fig = plt.subplot(3, 1, 1)
fig.set_title("Independent component $S_1$")
plt.plot(S[:, 0], color="#fc4f30")

fig = plt.subplot(3, 1, 2)
fig.set_title("Independent component $S_2$")
plt.plot(S[:, 1], color="#008fd5")

fig = plt.subplot(3, 1, 3)
fig.set_title("Independent component $S_3$")
plt.plot(S[:, 2], color="#e5ae38")
plt.tight_layout(pad=1)
plt.show()

A = np.array([[1, 1.2, 0.8], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
# Tạo dữ liệu
X = S @ A.T

fig = plt.subplot(3, 1, 1)
fig.set_title("The data after mixing $X_1$")
plt.plot(X[:, 0], color="#6d904f")

fig = plt.subplot(3, 1, 2)
fig.set_title("The data after mixing $X_2$")
plt.plot(X[:, 1], color="#8b8b8b")

fig = plt.subplot(3, 1, 3)
fig.set_title("The data after mixing $X_3$")
plt.plot(X[:, 2], color="#810f7c")
plt.tight_layout(pad=1)
plt.show()


def center_data(X):
    X_mean = X.T.mean(axis=0)
    X_new = X.T - X_mean
    return X_new.T


def whiten_data(X):
    Covar = np.cov(X)
    e, v = np.linalg.eigh(Covar + np.diag(np.full(Covar.shape[0], 0.0001)))
    return v @ np.diag(1.0 / np.sqrt(e)) @ v.T @ X


X_temp = X.T.copy()
print(X_temp.shape)

X_temp = center_data(X_temp)
print((X_temp.mean(axis=1)))

Y_temp = whiten_data(X_temp)
print(np.cov(Y_temp))

fig = plt.subplot(3, 1, 1)
fig.set_title("The data after centering and whitening Y_temp_1")
plt.plot(Y_temp[0, :], color="#6d904f")

fig = plt.subplot(3, 1, 2)
fig.set_title("The data after centering and whitening Y_temp_2")
plt.plot(Y_temp[1, :], color="#8b8b8b")

fig = plt.subplot(3, 1, 3)
fig.set_title("The data after centering and whitening Y_temp_3")
plt.plot(Y_temp[2, :], color="#810f7c")
plt.tight_layout(pad=1)
plt.show()

g = lambda x, alpha=1: np.tanh(alpha * x)
dg = lambda x, alpha=1: alpha * (1 - np.tanh(alpha * x) ** 2)


def fast_ICA_unit(Z, iteration=1000, tol=1e-6):
    w = np.random.rand(Z.shape[0], 1)
    w = w / np.linalg.norm(w)
    print((g(w.T @ Z) @ Z.T))
    for i in range(iteration):
        w_old = w.copy()
        w = (Z @ g(w.T @ Z).T) / Z.shape[1] - np.mean(dg(w.T @ Z)) * w
        # w = (g(w.T.dot(Z))@ Z.T)[0].reshape(-1,1)/Z.shape[1] - np.mean(dg(w.T@Z)) * w
        w = w / np.linalg.norm(w)
        if np.linalg.norm(w_old - w) < tol:
            print("number of iteration is : ", i)
            return w
    print("number of iteration is : ", iteration)
    return w


np.random.seed(None)
w = fast_ICA_unit(Y_temp)
print(w)
plot_signal(w, Y_temp)


def symetric_orthogonalization(W):
    U, S, Vt = np.linalg.svd(W.T, full_matrices=False)
    return Vt.T @ np.diag(1 / S) @ Vt @ W


def fast_ICA_multiple_IC(Z, n_component, iteration=1000, tol=1e-6):
    W = np.random.rand(Z.shape[0], n_component)

    W = symetric_orthogonalization(W)

    for i in range(iteration):
        # print(1)
        temp = W

        W = (Z @ g(W.T.dot(Z)).T) / Z.shape[1] - (
            np.sum(dg(W.T @ Z), axis=1) / Z.shape[1]
        ).reshape(1, -1) * W

        W = symetric_orthogonalization(W)

        if np.linalg.norm(temp - W) <= tol:
            print("number of iteration is : ", i)
            return W

    return W


W = fast_ICA_multiple_IC(Y_temp, 3, iteration=10000)
print(W)
plot_signal(W, Y_temp)
