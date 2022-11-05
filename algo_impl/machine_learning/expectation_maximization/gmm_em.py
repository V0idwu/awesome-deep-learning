#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@Time    :   2022/11/04 15:45:37
@Author  :   Tianyi Wu 
@Contact :   wutianyitower@hotmail.com
@File    :   gmm_em.py
@Version :   1.0
@Desc    :   None
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

plt.style.use("seaborn")


def gen_gmm_true_data(true_mu, true_var):
    # generate 3 clusters true data from 2-dim normal distribution
    num1, mu1, var1 = 400, true_mu[0], true_var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)

    num2, mu2, var2 = 600, true_mu[1], true_var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)

    num3, mu3, var3 = 1000, true_mu[2], true_var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)

    X = np.vstack((X1, X2, X3))

    # show data
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


def update_w(X, mu, var, pi):
    n_points, n_clusters = len(X), len(pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, mu[i], np.diag(var[i]))
    w = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return w


def update_pi(w):
    pi = w.sum(axis=0) / w.sum()
    return pi


def compute_log_likelyhood(X, pi, mu, var):
    n_points, n_clusters = len(X), len(pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, mu[i], np.diag(var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


def plot_clusters(X, mu, var, mu_true=None, var_true=None):
    colors = ["b", "g", "r"]
    n_clusters = len(mu)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {"fc": "None", "lw": 2, "edgecolor": colors[i], "ls": ":"}
        ellipse = Ellipse(mu[i], 3 * var[i][0], 3 * var[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (mu_true is not None) & (var_true is not None):
        for i in range(n_clusters):
            plot_args = {"fc": "None", "lw": 2, "edgecolor": colors[i], "alpha": 0.5}
            ellipse = Ellipse(mu_true[i], 3 * var_true[i][0], 3 * var_true[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()


def update_mu(X, w):
    n_clusters = w.shape[1]
    mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        mu[i] = np.average(X, axis=0, weights=w[:, i])
    return mu


def update_var(X, mu, w):
    n_clusters = w.shape[1]
    var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        var[i] = np.average((X - mu[i]) ** 2, axis=0, weights=w[:, i])
    return var


def expectation_opimization():
    # init
    true_mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_var = [[1, 3], [2, 2], [6, 2]]
    X = gen_gmm_true_data(true_mu, true_var)

    n_clusters = 3
    n_points = len(X)
    mu = [[0, -1], [6, 0], [0, 9]]
    var = [[1, 1], [1, 1], [1, 1]]
    pi = [1 / n_clusters] * 3
    w = np.ones((n_points, n_clusters)) / n_clusters
    pi = w.sum(axis=0) / w.sum()

    # update
    log_likelyhood_lst = []
    for i in range(5):
        plot_clusters(X, mu, var, true_mu, true_var)
        log_likelyhood_lst.append(compute_log_likelyhood(X, pi, mu, var))
        print("log-likehood:%.3f" % log_likelyhood_lst[-1])
        w = update_w(X, mu, var, pi)
        pi = update_pi(w)
        mu = update_mu(X, w)
        var = update_var(X, mu, w)


if __name__ == "__main__":

    expectation_opimization()
