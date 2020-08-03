from SVR import SVR
import numpy as np

X = np.loadtxt("./dataset_input.csv", delimiter=",")
y = np.loadtxt("./dataset_output.csv", delimiter=",")
y_0 = np.array([row[0] for row in y])
y_1 = np.array([row[1] for row in y])

n = np.size(y_1, 0)

print('ML test')

X_train, y_train = X[0:750], y_0[0:750]
X_test, y_test = X[750:n], y_0[750:n]

svr = SVR(gamma=0.1, verbose=True, max_iter=1)

print()
print(svr)
print()

svr.fit(X_train, y_train)
print()
print('mee score', svr.score(X_test, y_test))

print()
print('R2 score', svr._score(X_test, y_test, use_r2=True))
print()

print('Accuracy test')
print()

Xs = X[0:100]
ys = y_1[0:100]

svr = SVR(gamma=0.1, verbose=True, max_iter=1000, t=0)
K = svr._compute_kernel_matrix(Xs)

print('t = 0')
print()
svr._frank_wolfe_qp_solver_lin_pro(K, ys)
print()

svr = SVR(gamma=0.1, verbose=True, max_iter=1000, t=0.1)

print('t = 0.1')
print()
svr._frank_wolfe_qp_solver_lin_pro(K, ys)
print()

print('generic qp solver')
svr._generic_qp_solver(K, ys)
