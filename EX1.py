import torch
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0.3, 0.35, 0.28, 0.8, 0.7, 1, 0.8, 1.2, 1.6],
              [0.5, 0.45, 0.35, 0.75, 0.78, 0.7, 0.4, 0.5, 0.45]])
print(X)
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6]])
d1 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
d2 = np.array([0, 0, 0, 1, 1, 1, 1, 1 ,1])
n = 0.1
E1 = 0
E2 = 0
def step1(net1):
    if net1 >= 0:
        y1 = 1
    else: y1 = 0
    return y1
def step2(net2):
    if net2 >= 0:
        y2 = 1
    else: y2 = 0
    return y2
epoch = 0
converged = False  # Thêm biến kiểm tra hội tụ
while not converged:  # Chạy đến khi hội tụ
    E1, E2 = 0, 0  # Reset lỗi mỗi vòng lặp
    for i in range(len(d1)):
        net1 = W[:,0].T @ X[:,i]
        y1 = step1(net1)

        net2 = W[:,1].T @ X[:,i]
        y2 = step2(net2)

        W[:,0] += n * (d1[i] - y1) * X[:,i]
        W[:,1] += n * (d2[i] - y2) * X[:,i]

        E1 += 0.5 * ((d1[i] - y1) ** 2)
        E2 += 0.5 * ((d2[i] - y2) ** 2)

    epoch += 1
    print(f"Epoch {epoch}, E1: {E1}, E2: {E2}")

    # Kiểm tra điều kiện hội tụ
    if E1 == 0 and E2 == 0:
        converged = True  # Dừng vòng lặp nếu đã hội tụ

print(f"Huấn luyện hoàn tất sau {epoch} lần lặp! Trọng số cuối cùng:\n{W}")
X0 = np.ones((1,3))
X_input = np.random.rand(2,3)
X = np.vstack((X0, X_input))
W
Y = np.empty((2,3))
for i in range(len(X)):
    net1 = W[:,0].T @ X[:,i]
    y1 = step1(net1)
    Y[0,i] = y1

    net2 = W[:,1].T @ X[:,i]
    y2 = step2(net2)
    Y[1,i] = y2
print(f"X:{X}")
print(f"Y:{Y}")



