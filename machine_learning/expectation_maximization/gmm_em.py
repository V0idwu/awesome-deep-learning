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


class GMM:
    def __init__(self, sample_arr, target_mu_arr, target_var_arr) -> None:
        self.target_mu_arr, self.target_var_arr = target_mu_arr, target_var_arr
        self.x, self.n_clusters, self.n_samples = self.sample_points(sample_arr, target_mu_arr, target_var_arr)

    def sample_points(self, sample_arr, mu_arr, var_arr):
        n_clusters = len(sample_arr)
        n_samples = np.sum(sample_arr)

        self.train_x_lst = []
        for n_samples, mu, var in zip(sample_arr, mu_arr, var_arr):
            x = np.random.multivariate_normal(mu, np.diag(var), n_samples)
            self.train_x_lst.append(x)

        return np.vstack(self.train_x_lst), n_clusters, n_samples

    def show_data(self):
        plt.figure(figsize=(20, 12))
        # plt.axis([-10, 15, -5, 15])
        for x in self.train_x_lst:
            plt.scatter(x[:, 0], x[:, 1], s=15)
        plt.show()


class EM_GMM:
    def __init__(self, gmm) -> None:
        self.gmm = gmm
        self.x = gmm.x
        self.target_mu_arr, self.target_var_arr = gmm.target_mu_arr, gmm.target_var_arr
        self.mu = np.array([[0, -1], [6, 0], [0, 9]])
        self.var = np.array([[1, 1], [1, 1], [1, 1]])
        # self.mu = np.array([[1, 1], [1, 1], [1, 1]])
        # self.var = np.array([[1, 1], [1, 1], [1, 1]])
        self.w = np.ones((gmm.n_samples, gmm.n_clusters)) / gmm.n_clusters
        self.pi = self.compute_pi(self.w)

    def update_w(self, X, mu, var, pi):
        n_points, n_clusters = len(X), len(pi)
        pdfs = np.zeros(((n_points, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, mu[i], np.diag(var[i]))
        w = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return w

    @staticmethod
    def compute_pi(w):
        pi = w.sum(axis=0) / w.sum()
        return pi

    def update_mu(self, X, w):
        n_clusters = w.shape[1]
        mu = np.zeros((n_clusters, 2))
        for i in range(n_clusters):
            mu[i] = np.average(X, axis=0, weights=w[:, i])
        return mu

    def update_var(self, X, mu, w):
        n_clusters = w.shape[1]
        var = np.zeros((n_clusters, 2))
        for i in range(n_clusters):
            var[i] = np.average((X - mu[i]) ** 2, axis=0, weights=w[:, i])
        return var

    def em_process(self, n_iteration=10, verbose=False):

        # update
        log_likelyhood_lst = []
        for i in range(n_iteration):
            if verbose:
                self.plot_clusters(self.x, self.mu, self.var, self.target_mu_arr, self.target_var_arr)

            log_likelyhood_lst.append(self.compute_log_likelyhood(self.x, self.pi, self.mu, self.var))
            print("log-likehood:%.3f" % log_likelyhood_lst[-1])

            # E-step
            self.w = self.update_w(self.x, self.mu, self.var, self.pi)

            # M-step
            self.pi = self.compute_pi(self.w)
            self.mu = self.update_mu(self.x, self.w)
            self.var = self.update_var(self.x, self.mu, self.w)

        self.plot_clusters(self.x, self.mu, self.var, self.target_mu_arr, self.target_var_arr)

    @staticmethod
    def compute_log_likelyhood(x, pi, mu, var):
        n_samples, n_clusters = len(x), len(pi)
        pdfs = np.zeros(((n_samples, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = pi[i] * multivariate_normal.pdf(x, mu[i], np.diag(var[i]))
        return np.mean(np.log(pdfs.sum(axis=1)))

    @staticmethod
    def plot_clusters(X, mu, var, mu_true=None, var_true=None):
        colors = ["b", "g", "r"]
        n_clusters = len(mu)
        plt.figure(figsize=(20, 12))
        # plt.axis([-10, 15, -5, 15])
        plt.scatter(X[:, 0], X[:, 1], s=15)
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


if __name__ == "__main__":

    gmm = GMM(
        sample_arr=[400, 600, 1000], target_mu_arr=[[0.5, 0.5], [5.5, 2.5], [1, 7]], target_var_arr=[[1, 3], [2, 2], [6, 2]]
    )
    gmm.show_data()

    em_gmm = EM_GMM(gmm)
    em_gmm.em_process(n_iteration=1000, verbose=False)
