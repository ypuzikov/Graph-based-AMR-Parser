#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import cPickle as pickle
import logging
import time
import numpy as np
import pdb
import scipy.sparse as sp
import codecs

logger = logging.getLogger("main")


def rec_float():
    return collections.defaultdict(float)


def rec_int():
    return collections.defaultdict(int)

def split(A, cond):
    #http://stackoverflow.com/questions/7662458/how-to-split-an-array-according-to-a-condition-in-numpy
    return [A[cond], A[~cond]]

# https://gist.github.com/mblondel/656147
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


# https://github.com/ricefield/cs161sp12-linkpred/blob/master/python-perceptron/perceptron.py
def radial_basis_kernel(x, w, gamma):
    delta = np.linalg.norm(x - w)
    return np.exp(-gamma * delta * delta)


def polynomial_kernel(x, w, degree, alpha=1.0):
    return (np.dot(x, w) + alpha) ** degree


class SparsePerceptron(object):
    def __init__(self, n_feats=None, n_out=None):

        if n_feats is not None and n_out is not None:
            self.init_w(n_feats, n_out)

        else:
            self.n_out = None
            self.n_feat = None
            self.w = None
            self.train_steps = None
            self.train_totals = None

        self.i = 0


    # ==================================================
    # MULTICLASS CASE
    def init_w(self, n_feats, n_out):
        self.n_out = n_out
        self.n_feat = n_feats
        self.w = np.zeros((n_out, n_feats))
        self.train_steps = np.zeros((n_out, n_feats))
        self.train_totals = np.zeros((n_out, n_feats))

    def compute_scores(self, x_ind, x_dat):
        return self.w[:, x_ind].dot(x_dat)

    def update_one_w(self, clas, indices, data, eta):
        self.train_totals[clas, indices] += (self.i - self.train_steps[clas, indices]) * self.w[clas, indices]
        self.train_steps[clas, indices] = self.i
        self.w[clas, indices] += eta * data

    def update_weights(self, gold_label, predicted_label, x_ind, x_dat, eta=1.0):
        self.i += 1
        self.update_one_w(predicted_label, x_ind, x_dat, -eta)
        self.update_one_w(gold_label, x_ind, x_dat, eta)

    def avg_weights(self):
        """
        Perform weight averaging over all iterations
        Collins "Discriminative Training Methods" for Hidden Markov Models:..."
        """
        logger.debug("Averaging weights")
        self.train_totals += (self.i - self.train_steps) * self.w
        self.w = np.around(self.train_totals / float(self.i), 3)

    # ===============================================================

    def compute_scores_mat(self, x):
        return x.dot(self.w.T)

    def predict(self,x):
        scores = self.compute_scores_mat(x)
        return np.argmax(scores, axis=1)


    # def update_one_w(self, clas, indices, data, eta):
    #     self.train_totals[clas, indices] += (self.i - self.train_steps[clas, indices]) * self.w[clas, indices]
    #     self.train_steps[clas, indices] = self.i
    #     self.w[clas, indices] += eta * data
    #
    # def update_weights(self, gold_label, predicted_label, x_ind, x_dat, eta=1.0):
    #     self.i += 1
    #     self.update_one_w(predicted_label, x_ind, x_dat, -eta)
    #     self.update_one_w(gold_label, x_ind, x_dat, eta)
    #
    # def avg_weights(self):
    #     """
    #     Perform weight averaging over all iterations
    #     Collins "Discriminative Training Methods" for Hidden Markov Models:..."
    #     """
    #
    #     self.train_totals += (self.i - self.train_steps) * self.w
    #     self.w = np.around(self.train_totals / float(self.i), 3)

    # ==================================================
    # BINARY CASE
    def init_w_bi(self, n_feats):
        self.n_feat = n_feats
        self.w = np.zeros((n_feats))
        self.train_steps = np.zeros((n_feats))
        self.train_totals = np.zeros((n_feats))

    def compute_scores_bi(self, x_ind, x_dat):
        return self.w[x_ind].dot(x_dat)

    def update_weights_bi(self, indices, data, y, eta=1.0):
        self.i += 1
        self.train_totals[indices] += (self.i - self.train_steps[indices]) * self.w[indices]
        self.train_steps[indices] = self.i
        self.w[indices] += y * data * eta

    def avg_weights_bi(self):
        self.train_totals += (self.i - self.train_steps) * self.w
        self.w = np.around(self.train_totals / float(self.i), 3)

    # ==================================================

    def load(self, fname):
        logger.info("Loading the model from <----- %s " % (fname))
        with open(fname, "rb") as file_in:
            self.w = pickle.load(file_in)

    def save(self, model_fname):
        logger.info("Saving the model into -----> %s " % (model_fname))
        with open(model_fname, "wb") as file_out:
            pickle.dump(self.w, file_out)