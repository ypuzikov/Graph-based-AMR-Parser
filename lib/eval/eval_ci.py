#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from amrlib import AMR
from prf import compute_prf, compute_prf2


def extract_concepts_from_amrstring(amrstring):
    g = AMR(amrstring)
    constants = list(g.constants())
    concepts = [c._name for v, c in g.concepts()]
    all_concepts = constants + concepts

    return all_concepts


def eval_concepts(gold_concepts, pred_concepts):

    P, R, F1 = compute_prf2(gold_concepts, pred_concepts, stage="CI", save=False, savedir=None)
    print ("CI performance: %0.4f (P), %0.4f (R), %0.4f (F1)" % (P, R, F1))


def get_data_concepts_isi(fname):
    with open(fname) as infile:
        data_concepts = []

        for line in infile:

            if line.startswith("#") or not line.strip():
                continue

            amrstring = line.strip()
            datum_concepts = extract_concepts_from_amrstring(amrstring)
            data_concepts.append(datum_concepts)

        return data_concepts


def get_data_concepts_gold(fname):
    with open(fname) as infile:

        data_concepts = []
        amr_string = []
        inst = False

        for line in infile:
            if line.startswith("#"):
                continue

            elif not line.strip() and len(amr_string) > 0:
                amrstring = "".join(amr_string)
                datum_concepts = extract_concepts_from_amrstring(amrstring)
                data_concepts.append(datum_concepts)

                amr_string = []
                inst = False

            elif len(line) > 1:
                inst = True
                amr_string.append(line)

        if inst:
            amrstring = "".join(amr_string)
            datum_concepts = extract_concepts_from_amrstring(amrstring)
            data_concepts.append(datum_concepts)

        return data_concepts


def get_data_concepts_jamr(fname):
    """
    JAMR outputs amr strings, which parsimonious grammar
    doesn't like. JAMR output, however, stores concept predictions
    as attributes of nodes, so we can collect all of them,
    without using the parsimonious lib.

    """

    with open(fname) as infile:

        data_concepts = []
        datum_concepts = []

        for line in infile:
            if line.startswith("# ::node"):
                items = line.strip().split()
                concept = items[3]
                datum_concepts.append(concept)

            elif not line.strip() and len(datum_concepts) > 0:
                data_concepts.append(datum_concepts)
                datum_concepts = []

            else:
                continue

        return data_concepts

def parse_args():
    import argparse

    usage = "python eval_ci.py file1 file2 --fmt format_of_the_file_with_predicted_amrs"
    arg_parser = argparse.ArgumentParser(usage=usage)
    arg_parser.add_argument("input", nargs=2)
    arg_parser.add_argument("--fmt", choices=["gold", "isi", "jamr"], required=True)

    args = arg_parser.parse_args()
    return args


def main(args):
    gold_fname, pred_fname = args.input
    pred_fmt = args.fmt

    gold_concepts = get_data_concepts_gold(gold_fname)

    if pred_fmt == "gold":
        pred_concepts = get_data_concepts_gold(pred_fname)

    elif pred_fmt == "isi":
        pred_concepts = get_data_concepts_isi(pred_fname)

    else:
        pred_concepts = get_data_concepts_jamr(pred_fname)

    eval_concepts(gold_concepts, pred_concepts)


if __name__ == "__main__":
    args = parse_args()
    main(args)
