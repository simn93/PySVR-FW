# coding=utf-8
#
# ======================================================================================================================
# Computational Mathematics for Learning and Data Analysis
# ----------------------------------------------------------------------------------------------------------------------
# [ML Projects] Project 13
# ----------------------------------------------------------------------------------------------------------------------
# Participate to the ML cup competition associated with the Machine Learning course
# https://elearning.di.unipi.it/mod/folder/view.php?id=3615
# with a SVR-type approach of your choice (in particular, with one or more kernels of your choice).
# Implement yourself a training algorithm of the SVR using a Frank-Wolfe type (conditional gradient)
# [references: https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm], using the programming language of your
# choice (C/C++, Python, Matlab) but no ready-to-use optimization libraries.
# For the avoidance of doubt, this means that you may use library functions (Matlab ones or otherwise) if an inner step
# of the algorithm requires them as a subroutine (for instance, for solving the LP within the algorithm - but do
# consider developing ad-hoc approaches exploiting the structure of your problem), but your final implementation should
# not be a single library call.
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
#  Authors:
#       Carmine Caserio
#       Simone Schirinzi
#  Date: 18-04-19
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================

import numpy as np
import matlab.engine
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import r2_score
import time


class SVR(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf', gamma=0.01, kernel_matrix=None, tol=1e-3, C=1.0,
                 epsilon=0.1, verbose=False, max_iter=200, t=0.1, use_bias=False):
        """
        This class represents the implementation of an SVR using Frank-Wolfe for solving quadratic optimization.

        :param kernel: SVR kernel.
        Support 'rbf'.
        :param gamma: Rbf kernel parameter
        :param tol: Frank-Wolfe solution tolerance
        :param C: SVR data-point tolerance
        :param epsilon: SVR epsilon tube
        :param verbose: True or False
        :param max_iter: Frank-Wolfe max iter
        :param t: Frank-Wolfe ball regularization
        :param kernel_matrix: precomputed kernel matrix
        :param use_bias: when True compute bias
        """
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.verbose = verbose
        self.max_iter = max_iter
        self.t = t
        self.kernel_matrix = kernel_matrix
        self.use_bias = use_bias

        # Private variable
        self._kernel_function = self._name_to_function(self.kernel)
        self._fit_beta = None
        self._fit_X = None
        self._fit_b = None

    def fit(self, X, y, use_fw=True):
        """
        It computes lagrangian multiplier using Frank-Wolfe Quasi-Newton method.
        :param X: Dataset input as matrix
        :param y: Dataset output as matrix
        :param use_fw: if True use Frank-Wolfe as quadratic optimization solver, else use generic one
        :return: trained model
        """

        if type(y) != np.ndarray:
            y = np.array(y)
            if self.verbose:
                print("numpy array required for output matrix, conversion executed")

        if type(X) != np.ndarray:
            X = np.array(X)
            if self.verbose:
                print("numpy array required for input matrix, conversion executed")

        K = self.kernel_matrix if self.kernel_matrix is not None else self._compute_kernel_matrix(X)
        beta = self._frank_wolfe_qp_solver_lin_pro(K, y) if use_fw else self._generic_qp_solver(K, y)
        b = self._compute_bias(beta, K, y) if self.use_bias else 0

        self._fit_beta = beta
        self._fit_X = X
        self._fit_b = b

        return self

    def predict(self, X):
        """
        It predicts value using a trained model.
        :param X: data to predict
        :return: y value vector predicted
        """

        assert self._fit_beta is not None, """ Model not trained "Exception"  """

        y = np.zeros(np.size(X, 0))

        for idx, data in enumerate(X):
            for i in range(np.size(self._fit_X, 0)):
                y[idx] += self._fit_beta[i] * self._kernel_function(self._fit_X[i], data) + self._fit_b

        return y

    def score(self, X, y, sample_weight=None):
        """
        Score a trained model using mee score.
        :param X: Test set
        :param y: Test label
        :param sample_weight: Sample Weight
        :return: mee score
        """
        return self._score(X, y, None, False)

    def _score(self, X, y, sample_weight=None, use_r2=False):
        """
        Score a trained model using mee or r2 score.
        :param X: Test set
        :param y: Test label
        :param sample_weight: Sample Weight
        :param use_r2: if True use mee score function, else use mee
        :return: r2 score
        """
        y_true = np.array(y)
        y_pred = np.array(self.predict(X))

        return self._score_function(y_true, y_pred, use_r2)

    @staticmethod
    def _score_function(y_true, y_pred, use_r2=False):
        if use_r2:
            return r2_score(y_true, y_pred)
        else:
            ret = np.sqrt(np.power(np.subtract(y_true, y_pred), 2))
            return np.sum(ret) / len(ret)

    # --------------------- Kernel zone ---------------------
    def _radial_basis_kernel(self, x, xi):
        ret_val = np.linalg.norm(x - xi, 2)
        ret_val = np.exp(-self.gamma * np.power(ret_val, 2))
        return ret_val

    def _name_to_function(self, kernel_func):
        if kernel_func == 'rbf':
            return self._radial_basis_kernel

    def _compute_kernel_matrix(self, inputs):
        n = inputs.shape[0]
        k = np.zeros(shape=(n, n))

        for i, x_i in enumerate(inputs):
            for j, x_j in enumerate(inputs):
                k[i, j] = self._kernel_function(x_i, x_j)
        return k

    # --------------------- SVR zone ---------------------

    def _compute_bias(self, beta, K, y):
        """
        It finds bias from the average of support vectors with interpolation error e.
        SVs with interpolation error e have alphas: 0 < alpha < C.
        :param beta: vector of beta's
        :param K: Kernel matrix
        :param y: prediction output
        :return: bias value
        """

        b_sum = 0.0
        b_sum_count = 0
        for idx, b_i in enumerate(beta):
            if self.epsilon < abs(b_i) < (self.C - self.epsilon):
                b_sum_count += 1
                b_sum += y[idx] - self.epsilon * np.sign(b_i) - np.dot(beta, K[idx])

        return (b_sum / b_sum_count) if b_sum_count > 0 else ((max(y) + min(y)) / 2)

    # --------------------- Frank-Wolfe zone ---------------------

    def _frank_wolfe_qp_solver_lin_pro(self, K, y):
        """
        It is the Frank-Wolfe minimization algorithm solver for SVR quadratic optimization.
        Require: K, y as numpy array (else some operations are mis-defined).

        :param K: kernel matrix
        :param y: dataset value
        :return: lagrangian's alpha
        """

        C = self.C
        eps_svr = self.epsilon
        eps_fw = self.tol
        max_iter = self.max_iter
        t = self.t
        linprog_tol = self.tol

        n = len(y)
        eps_vec = eps_svr * np.ones(n)

        # Initial eligible point: 0
        # Satisfies the two constraints trivially
        beta = np.zeros(n)
        gamma = np.zeros(n)

        iteration = 1
        best_lb = -np.inf

        # Solve min{ f(beta, gamma) } in {-C <= beta <= C and sum(beta)=0 and gamma >= -beta and gamma >= -beta}
        # with f(beta, gamma) = 1/2 * beta * K * beta - y * beta + eps_svr * gamma
        #
        # with gradient of f(beta, gamma) = [ K * beta - y, eps_svr ]

        # vector of constant bound for linear programming
        eye_plus = [[0] * i + [1] + [0] * (n - i - 1) for i in range(n)]
        eye_minus = [[0] * i + [-1] + [0] * (n - i - 1) for i in range(n)]

        A_ub = np.concatenate((
            np.concatenate((eye_plus, eye_minus), axis=1),
            np.concatenate((eye_minus, eye_minus), axis=1)
        ))
        b_ub = np.array([0] * 2 * n)
        A_eq = np.array([[1] * n + [0] * n])
        b_eq = np.array([0])
        bounds = np.array([(-C, +C)] * n + [(None, None)] * n)

        if self.verbose:
            print('iter', 'value', 'best_lb', 'gap', 'alpha', 'time', sep=',')

        while True:
            start_time = time.time()

            v = 0.5 * np.dot(np.dot(beta, K), beta) - np.dot(y, beta) + np.dot(eps_vec, gamma)
            g_beta = np.dot(K, beta) - y
            g_gamma = eps_vec

            # solve z1, z2 =  argmin{ g_beta * z_beta + g_gamma * z_gamma :
            #                           -C <= z1 <= C
            #                           \sum_{i=0}^n z_beta_i = 0
            #                           z_gamma \geq z_beta
            #                           z_gamma \geq -z_beta
            # using linear programming treating the two variable vector as ones of size 2*n

            c = np.concatenate((g_beta, g_gamma))

            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='interior-point',
                          options={'presolve': False, 'disp': False, 'tol': linprog_tol})
            z_beta = np.array(res['x'][0:n])
            z_gamma = np.array(res['x'][n:2 * n])

            # lower bound using first-order approximation: f(x) + g(y-x)
            # and termination checking
            lb = v + np.dot(g_beta, z_beta - beta) + np.dot(g_gamma, z_gamma - gamma)
            best_lb = lb if lb > best_lb else best_lb

            gap = (v - best_lb) / max([abs(v), 1])

            # termination checking
            if gap <= eps_fw:
                if self.verbose:
                    print("finished with gap", gap, "<=", eps_fw)
                return beta

            # FW stabilization
            if t > 0:
                bounds_t = np.array(
                    [(max(-C, b_i - t), min(C, b_i + t)) for b_i in beta] +
                    [(g_i - t, g_i + t) for g_i in gamma]
                )
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds_t, method='interior-point',
                              options={'presolve': False, 'disp': False, 'tol': linprog_tol})
                z_beta = np.array(res['x'][0:n])
                z_gamma = np.array(res['x'][n:2 * n])

            # direction
            d_beta = z_beta - beta
            d_gamma = z_gamma - gamma

            # compute optimal step:
            # min{   f( beta + a * d_beta, gamma + a * d_gamma) }
            # subject to 0 <= alpha <= 1
            #
            #  (1 / 2)(beta + a*d_beta )*K*( beta + a*d_beta ) - y*(beta + a*d_beta) + eps_vec*(gamma + a*d_gamma)
            #  (1 / 2) a^2 (d_beta' * K * d_beta) + a( d_beta' * K * beta - d_beta' * y + eps_svr * d_gamma)[+ cost]
            #
            #  == > a * d_beta' * K * d_beta + ( d_beta' * K * beta - d_beta' * y + eps_svr * d_gamma ) = 0
            #  == > a = ( d_beta' * ( y - K * beta ) - d_gamma' * eps_svr ) / d_beta' * K * d_beta

            num: float = np.dot(-g_beta, d_beta) + np.dot(-g_gamma, d_gamma)
            den: float = np.dot(np.dot(d_beta, K), d_beta)

            if den <= 1e-16:
                alpha = 1
            else:
                alpha = max(0.0, min(1.0, num / den))

            if alpha == 0:
                print("finished why alpha is zero:", num / den, "< 0")
                return beta

            beta += alpha * d_beta
            gamma += alpha * d_gamma

            stop_time = time.time()
            iter_time = stop_time - start_time

            # statistic print
            if self.verbose:
                print(iteration, v, best_lb, gap, alpha, iter_time, sep=',')

            iteration += 1
            if iteration > max_iter:
                if self.verbose:
                    print("finished with max", max_iter, "iterations")
                return beta

    def _generic_qp_solver(self, K, y):
        """
        It is the general algorithm for SVR quadratic optimization.
        Require: K, y as numpy array (else some operations are mis-defined).

        :param K: kernel matrix
        :param y: dataset value
        :return: lagrangian's alpha
        """

        n = len(y)

        eye_plus = [[0.0] * i + [1.0] + [0.0] * (n - i - 1) for i in range(n)]
        eye_minus = [[0.0] * i + [-1.0] + [0.0] * (n - i - 1) for i in range(n)]
        zeros = [[0.0] * n for i in range(n)]

        P = np.concatenate((
            np.concatenate((K, zeros), axis=1),
            np.concatenate((zeros, zeros), axis=1)
        ))

        q = np.concatenate((-y, [float(self.epsilon)] * n))

        G = np.concatenate((
            np.concatenate((eye_plus, zeros), axis=1),
            np.concatenate((eye_minus, zeros), axis=1),
            np.concatenate((eye_plus, eye_minus), axis=1),
            np.concatenate((eye_minus, eye_minus), axis=1)
        ))

        h = np.array([float(self.C)] * n * 2 + [0.0] * n * 2)
        A = np.array([[1.0] * n + [0.0] * n])
        b = np.array([0.0])

        m_H = matlab.double(P.tolist())
        m_f = matlab.double(q.tolist())
        m_A = matlab.double(G.tolist())
        m_b = matlab.double(h.tolist())
        m_Aeq = matlab.double(A.tolist())
        m_beq = matlab.double(b.tolist())

        eng = matlab.engine.start_matlab()
        start_time = time.time()
        res = eng.quadprog(m_H, m_f, m_A, m_b, m_Aeq, m_beq)
        stop_time = time.time()

        if self.verbose:
            print('time:', str(stop_time - start_time))

        beta = res[0:n]

        return beta
