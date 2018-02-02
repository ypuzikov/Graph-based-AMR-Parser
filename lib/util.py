#!/usr/bin/env python
# -*- coding: utf-8


import os, sys
import collections
import numpy as np
import scipy.sparse as ssp
import cPickle as pickle
import logging
import matplotlib.pyplot as plt

def rec_dd():
    return collections.defaultdict()

def rec_ddd():
    return collections.defaultdict(rec_dd)

def iterate_over_corpus(corpus):
    for sentence in corpus:
        yield sentence

# ==================================== MATH

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return ssp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def flatten(lst):
    """
    Returns a total sum over an array with values being int, arrays or matrices
    :param lst: array to be flattened and summed over
    :return: one scalar number - sum
    """
    return sum(([x] if not isinstance(x, list) else flatten(x)
                for x in lst), [])

def split_array(arr, cond):
    return [arr[cond], arr[~cond]]

def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

# ==================================== PLOTTING

def plot_cm(conf_arr, labels):

    """
    conf_arr = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
                [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
                [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]


    """

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(11,11))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,interpolation='nearest') # cmap=plt.cm.Blues

    width, height = conf_arr.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    # kwargs are needed in order
    # to match the size of the colorbar
    # with that of a confusion matrix
    fig.colorbar(res,fraction=0.046, pad=0.04)
    ticks = np.arange(width)
    plt.xticks(ticks, labels, rotation=45, fontsize=15)
    plt.yticks(ticks, labels, fontsize=15)
    plt.xlabel("Predicted labels", fontsize=20)
    plt.ylabel("True labels", fontsize=20)
    plt.tight_layout()
    ax.tick_params(labelsize=15)

    plt.show()
    fig.savefig('../stats/confusion_matrix.pdf')


def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()
        label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                "%0.2f" % float(height),
                ha="center", va="bottom", fontsize=15)

def plot_prf(A=(0.82, 0.72, 0.76), B=(0.86, 0.75, 0.80)):

    # ci_jamr = [0.82, 0.72, 0.76]
    # ci_pamr = [0.86, 0.75, 0.80]
    # ci_name = "Concept identification"
    #
    # full_jamr = [0.64, 0.48, 0.55]
    # full_pamr = [0.5, 0.56, 0.53]
    # full_title = "Full parsing performance"
    #
    # speed_jamr = [1, 2, 3]
    # speed_pamr = [1, 1, 2]
    # speed_name = "Speed comparison"

    # create plot
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)

    bar_width = 0.35
    opacity = 0.3
    index = np.arange(3)

    rect1 = ax1.bar(index, A, bar_width, alpha=opacity, color="b", label="JAMR")
    rect2 = ax1.bar(index + bar_width, B, bar_width, alpha=opacity, color="g", label="Proposed")
    # ax1.set_title("Concept identification")

    plt.xticks(index + bar_width, ("P", "R", "F1"), fontsize=20)
    ax1.set_ylim([0.4, 1.0])
    ax1.tick_params(labelsize=18)
    plt.ylabel("Scores", fontsize=20)
    plt.legend()

    autolabel(rect1, ax1)
    autolabel(rect2, ax1)

    # plt.title("Parsing performance")

    plt.show()
    fig.savefig("dummy_performance.pdf")

# ============================================ MISC

def load_pkl(fname):
    with open(fname) as infile:
        return pickle.load(infile)

def get_fname(base_fn):
    # get a filename, resolving name collisions
    id = 0
    new_base_fn = base_fn
    while os.path.exists(new_base_fn):
        new_base_fn = base_fn
        id += 1
        new_base_fn += "_" + str(id)

    return new_base_fn

def init_log(filename):
    # global logger
    logger = logging.getLogger("main")
    fmt = "%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s"
    fmt_date = "%Y-%m-%dT%T%Z"
    formatter = logging.Formatter(fmt, fmt_date)

    id = 0
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = filename
        id += 1
        new_filename += "_" + str(id)

    f_hdlr = logging.FileHandler(new_filename)
    f_hdlr.setLevel(logging.DEBUG)
    f_hdlr.setFormatter(formatter)
    logger.addHandler(f_hdlr)

    out_hdlr = logging.StreamHandler(sys.stdout)
    out_hdlr.setLevel(logging.INFO)
    out_hdlr.setFormatter(formatter)
    logger.addHandler(out_hdlr)
    logger.setLevel(logging.DEBUG)
    return logger

def gen_cvb_db(db_fname):
    """ Create a DB from CVB data"""

    CVB = open(db_fname, "r")
    N_dict = collections.defaultdict(rec_dd)
    AJ_dict = collections.defaultdict(rec_dd)
    AV_dict = collections.defaultdict(rec_dd)
    V_dict = collections.defaultdict(rec_dd)

    cvb_dict = {"N": N_dict,
                "AJ": AJ_dict,
                "AV": AV_dict,
                "V": V_dict}

    for idx, line in enumerate(CVB):
        print "Processed line %d" % (idx)
        elems = [tuple(pair.split("_")) for pair in line.strip().split("#")]
        if len(elems) == 0:
            continue
        for e in elems:  # we have tuples (word1, pos1), (word2,pos2), etc.
            others = [el for el in elems if el != e]  # get other tuples
            for other_pair in others:
                cvb_dict[e[1]].keys().append(e[0])
                cvb_dict[e[1]][e[0]].setdefault(other_pair[1], []).append(other_pair[0])

    return cvb_dict

