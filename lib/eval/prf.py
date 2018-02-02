#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, re
import logging
import json

logger = logging.getLogger("main")

def compute_prf(Y_true, Y_pred, stage, save=False, savedir=None):

    """
    Computes precision, recall, f-measure for sets, created from lists.
    That is, consider P, R, F, computed on ** unique ** elements of two lists
    :param Y_true: a list, containing gold labels - each is a list of items itself
    :param Y_pred: a list, containing predicted labels - each is a list of items itself
    :param stage: either CI - concept identification, or RI - relation identification
    :param save: boolean, whether to save comparison table or not
    :param savedir: directory where the comparison table should be stored

    """

    len_true = 0
    len_pred = 0
    len_i = 0
    assert len(Y_true) == len(Y_pred)

    comparison = []

    for idx, y_true in enumerate(Y_true):
        y_pred = Y_pred[idx]

        y_pred_set = set(y_pred)
        y_true_set = set(y_true)
        i = set.intersection(y_pred_set, y_true_set)
        len_true += len(y_true_set)
        len_pred += len(y_pred_set)
        len_i += len(i)

        oneR = len(i) / float(len(y_true)) if len(y_true) > 0 else 0.0
        oneP = len(i) / float(len(y_pred)) if len(y_pred) > 0 else 0.0
        oneF = 2 * (oneP * oneR) / (oneP + oneR) if (oneP + oneR > 0.0) else 0.0
        comparison.append(("%f" % (oneP), "%f" % (oneR), "%f" % (oneF), tuple(y_pred), tuple(y_true)))

    P = len_i / float(len_pred) if len_pred > 0 else 0.0
    R = len_i / float(len_true) if len_true > 0 else 0.0
    F1 = 2 * (P * R) / (P + R) if (P + R > 0.0) else 0.0

    if save:
        assert savedir is not None
        cfn = os.path.join(savedir, "%s_errors.json" %(stage))
        json.dump(comparison, open(cfn, "w"))
        logger.info("Saved error table to --> %s" %(cfn))

    return P, R, F1

def rename_duplicates(L):
    newlist = []

    for i, v in enumerate(L):
        totalcount = L.count(v)
        count = L[:i].count(v)
        newlist.append(v + str(count + 1) if totalcount > 1 else v)

    return newlist

def compute_prf2(Y_true, Y_pred, stage, save=False, savedir=None):

    """
    Computes precision, recall, f-measure for lists
    That is, consider P, R, F, computed on ** all ** elements of two lists
    :param Y_true: a list, containing gold labels - each is a list of items itself
    :param Y_pred: a list, containing predicted labels - each is a list of items itself
    :param stage: either CI - concept identification, or RI - relation identification
    :param save: boolean, whether to save comparison table or not
    :param savedir: directory where the comparison table should be stored
    """

    len_true = 0
    len_pred = 0
    len_i = 0
    assert len(Y_true) == len(Y_pred)

    comparison = []

    for idx, y_true in enumerate(Y_true):
        y_pred = Y_pred[idx]

        y_pred_renamed = rename_duplicates(y_pred)
        y_true_renamed = rename_duplicates(y_true)

        i = [c for c in y_pred_renamed if c in y_true_renamed]
        len_true += len(y_true_renamed)
        len_pred += len(y_pred_renamed)
        len_i += len(i)

        oneR = len(i) / float(len(y_true_renamed)) if len(y_true_renamed) > 0 else 0.0
        oneP = len(i) / float(len(y_pred_renamed)) if len(y_pred_renamed) > 0 else 0.0
        oneF = 2 * (oneP * oneR) / (oneP + oneR) if (oneP + oneR > 0.0) else 0.0
        comparison.append(("%f" % (oneP), "%f" % (oneR), "%f" % (oneF), tuple(y_pred_renamed), tuple(y_true_renamed)))

    P = len_i / float(len_pred) if len_pred > 0 else 0.0
    R = len_i / float(len_true) if len_true > 0 else 0.0
    F1 = 2 * (P * R) / (P + R) if (P + R > 0.0) else 0.0

    if save:
        assert savedir is not None
        cfn = os.path.join(savedir, "%s_errors.json" %(stage))
        json.dump(comparison, open(cfn, "w"))
        logger.info("Saved error table to --> %s" %(cfn))

    return P, R, F1


def eval_ISI_align(pred_file, gold_file):


    """

    Computes precision, recall, f-measure for the ISI alignment.
    Defined similar to the prf function above, but not storing
    the comparison table.

    """

    gold_A = []
    test_A = []
    gold_A_no_roles = []
    test_A_no_roles = []

    align_pat = re.compile("(# ::alignments\s)(.+)(\n)")

    with open(pred_file) as test:
        for line in test:
            A = []
            A_no_roles = []
            a_pairs = line.strip().split()
            for pair in a_pairs:
                a = pair.split("-")
                if a[1][-1] != "r":
                    A_no_roles.append(tuple(a))
                A.append(tuple(a))
            test_A.append(A)
            test_A_no_roles.append(A_no_roles)

    with open(gold_file) as gold:
        for line in gold:
            if align_pat.match(line):
                A = []
                A_no_roles = []
                alignments = align_pat.match(line).group(2)
                a_pairs = alignments.split()
                for pair in a_pairs:
                    a = pair.split("-")
                    if a[1][-1] != "r":
                        A_no_roles.append(tuple(a))
                    A.append(tuple(a))
                gold_A.append(A)
                gold_A_no_roles.append(A_no_roles)

    t_count = len(test_A)
    g_count = len(gold_A)

    assert g_count == t_count, "Total counts: golden - %d, predicted -%d. Mismatch!" \
                               % (g_count, t_count)

    len_g = len_p = len_i = nr_len_g = nr_len_p = nr_len_i = 0.0

    for idx, test_al in enumerate(test_A):
        # w/ role tags
        golden_alignments = set(gold_A[idx])
        predicted_alignments = set(test_al)

        len_g += len(golden_alignments)
        len_p += len(predicted_alignments)
        len_i += len(golden_alignments & predicted_alignments)

        # w/o role tags
        nr_golden_alignments = set(gold_A_no_roles[idx])
        nr_predicted_alignments = set(test_A_no_roles[idx])

        nr_len_g += len(nr_golden_alignments)
        nr_len_p += len(nr_predicted_alignments)
        nr_len_i += len(nr_golden_alignments & nr_predicted_alignments)

    pr = len_i / len_p if len_p > 0 else 0
    r = len_i / len_g if len_g > 0 else 0
    f = 2 * (pr * r) / (pr + r)

    nr_pr = nr_len_i / nr_len_p if nr_len_p > 0 else 0
    nr_r = nr_len_i / nr_len_g if nr_len_g > 0 else 0
    nr_f = 2 * (nr_pr * nr_r) / (nr_pr + nr_r)

    print "Stats w/ role tokens:\nPrecision: %f\nRecall: %f\nF-score: %f\n\n" \
          "Stats w/o role tokens:\nPrecision: %f\nRecall: %f\nF-score: %f" \
          % (pr, r, f, nr_pr, nr_r, nr_f)


if __name__ == "__main__":
    argvs = sys.argv
    assert len(argvs) == 3, "Need 2 files as input: predicted alignments, golden alignments"
    eval_ISI_align(argvs[1], argvs[2])
