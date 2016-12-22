import numpy as np

#活性化関数にシグモイド関数を使用(h)
def sigmoid(x):
    return 1/(1+np.exp(-x))

#恒等関数(σ)
def identity_function(x):
    return x

#0層目〜1層目
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

Z1 = sigmoid(A1)
print(Z1)
print("\n")

#1層目〜2層目
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print(A2)

Z2 = sigmoid(A2)
print(Z2)
print("\n")

#2層目〜3層目
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print(Z2.shape)
print(W3.shape)
print(B3.shape)

A3 = np.dot(Z2, W3) + B3
print(A3)

Y = identity_function(A3)
print(Y)
