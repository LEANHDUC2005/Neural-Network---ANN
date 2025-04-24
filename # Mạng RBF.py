import numpy as np
import math
import os
os.rename("# Mạng RBF.py", "Mạng RBG - Bài toán nhận diện công logic XOR.py")

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
d = np.array([0, 1, 1, 0])
n = 0.1

u_1 = X[:, 3].reshape(2,1)
u_2 = X[:, 0].reshape(2,1)
u = np.hstack((u_1, u_2))
W = np.random.rand(3)

def khoangcach(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def sigma(x2, x1, y2, y1):
    return float( khoangcach(x2, x1, y2, y1) / math.sqrt(2) )

def RBF(a, b, c, d, e):
    if e > 0:
        return math.exp(-((khoangcach(a, b, c, d)**2) / ((e**2))))
    return 0.0

S = sigma(int(u[0,0]), int(u[0,1]), int(u[1,0]), int(u[1,1]))

fi = np.empty((2, 4))
for i in range(X.shape[1]):
    for j in range(2):
        fi[j, i] = round(RBF(int(X[0, i]), int(u[0, j]), int(X[1, i]), int(u[1, j]), S),3)
fi = np.vstack((np.ones((1,X.shape[1])),fi))
# Huấn luyện
Huanluyen = 0
Hoitu = False
MAX_EPOCH = 1000

while not Hoitu and Huanluyen < MAX_EPOCH:
    E = 0
    for i in range(X.shape[1]):
        net = np.dot(W, fi[:, i])
        y = net
        e = d[i] - y
        W = W + n * e * fi[:, i]
        E += 0.5 * (e ** 2)
    if Huanluyen % 100 == 0:
        print(f"Epoch {Huanluyen}, Error = {E:.5f}")
    if E <= 1e-2:
        Hoitu = True
    Huanluyen += 1

print(f"\nSố lần huấn luyện: {Huanluyen}")
print(f"Sai số cuối cùng: {E:.5f}")
print(f"Trọng số cuối cùng W: {W}")
print(f"Đầu vào RBF: {fi}")
print(f"Dolechchuan: {S})")
print("\nDự đoán sau huấn luyện:")
for i in range(X.shape[1]):
    net = np.dot(W, fi[:, i])
    y = 1 if net >= 0.5 else 0
    print(f"Input: {X[:, i]} => Output: {y} (Target: {d[i]})")


    

    
        