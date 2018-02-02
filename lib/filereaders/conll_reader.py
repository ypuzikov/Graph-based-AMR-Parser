#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import csv
import logging
import networkx as nx

from base_reader import FileReader
from lib.util import rec_dd, rec_ddd

logger = logging.getLogger("main")

class ConllFileReader(FileReader):

    def get_sbls(self, tokid, D):
        headid = D.predecessors(tokid)

        if len(headid) == 0:  # no predecessors (root)
            return [], []
        # dpg is a tree => only one predecessor possible (head)
        sbls = D.successors(headid[0])
        rsbs = []
        lsbs = []

        for s in sorted(sbls):
            if s < tokid:
                lsbs.append(s)
            else:
                rsbs.append(s)

        return lsbs, rsbs

    def get_paths(self, depG):
        """
        Compute shortest paths between each pair of nodes in the graph.
        Returns a dictionary of the form {a: {a:[a], b:[a,c,b], ...}}
        Note: includes a self-path.

        """

        allpaths = {}
        for node, node_ats in depG.nodes(data=True):
            allpaths[node] = nx.shortest_path(depG, source=node)

        return allpaths

    def init_datum(self):

        return {"id": None, "tokens": [], "comments": {},
                "depG": None, "amrG": None, "spaths": None, "ri_data": collections.defaultdict(rec_ddd),
                # "amrP": None,
                }

    def process_newline(self, curr_datum, dG, h2m, data, pas):

        # add all extracted edges
        dG.add_edges_from(ebunch=h2m)

        preds, args = pas

        # fill sbl info
        for tokid, tok in enumerate(curr_datum["tokens"]):
            lsb, rsb = self.get_sbls(tokid, dG)
            tok["lsb"] = lsb
            tok["rsb"] = rsb
            tok["preds"] = preds.get(tokid, None)
            tok["args"] = args.get(tokid, None)

        # finalize the datum
        synpaths = self.get_paths(dG)
        curr_datum["depG"] = dG
        curr_datum["spaths"] = synpaths
        curr_datum["comments"]["snt"] = " ".join([T["word"] for T in curr_datum["tokens"]])
        curr_datum["comments"]["tok"] = curr_datum["comments"]["snt"]
        curr_datum["comments"]["annotator"] = "pamr"
        data.append(curr_datum)


class ConllTrainFileReader(ConllFileReader):

    def read_row(self, curr_datum, row, dG, h2m):

        position = int(row[0]) - 1
        head_position = int(row[5]) - 1

        word = row[1]
        lem = row[2]
        pos = row[3]
        ne = row[4]
        deprel = row[6]

        if row[11] == "Y":
            items = row[12].split(".")
            if len(items) == 2:
                sense = int(items[1])
                ispred = True

            else:
                # this happens on urls
                logger.info("Strange predicate: %s" % row[12])
                ispred = False  # fixme
                sense = None
                pass

        else:
            ispred = False
            sense = None

        token = {"word": word, "lemma": lem, "pos": pos, "ne": ne, "rel": deprel,
                 "ispred": ispred, "sense": sense, "preds": None, "args": None,
                 "head": head_position, "lsb": None, "rsb": None,
                 "netype": None, "ctype": None, "concepts": [None], "neg": False, "node_id": [], "ci_data": (),

                 }

        self.vocab.append(word.lower())
        self.vocab.append(lem)
        self.vocab.append(pos)
        self.vocab.append(ne)
        self.vocab.append(deprel)

        dG.add_node(position, attr_dict=token)

        if deprel != "ROOT":
            # (parent, child, role) in the dependency tree
            h2m.append((head_position, position, {"label": token["rel"]}))
        else:
            dG.graph["root"] = int(row[0]) - 1

        curr_datum["tokens"].append(token)

    def extract_semroles(self):

        voc = []
        assert type(self.vocab) == list

        with open(self.fname, "r") as conll_result:

            reader = csv.reader(conll_result, delimiter="\t")

            all_pas = []
            pred_idx = collections.defaultdict()
            pred_counter = 0
            pas_triples = []

            for row in reader:
                if row[0].isdigit():
                    tokid = int(row[0]) - 1

                    if row[11] == "Y":
                        pred_idx[pred_counter] = tokid
                        pred_counter += 1

                    for col_idx, col_data in enumerate(row[13:]):
                        if col_data != "_":
                            pas_triples.append((tokid, col_idx, col_data))

                else:
                    # EOS
                    preds = collections.defaultdict(rec_dd)
                    args = collections.defaultdict(rec_dd)

                    for triple in pas_triples:
                        tokid, pred_num, role = triple
                        pred_tokid = pred_idx[pred_num]
                        preds.setdefault(pred_tokid, []).append((tokid, role))
                        args.setdefault(tokid, []).append((pred_tokid, role))
                        voc.append(role)

                    all_pas.append((preds, args))

                    pred_idx = {}
                    pred_counter = 0
                    pas_triples = []

            voc = list(set(voc))
            self.vocab.extend(voc)

            return all_pas

    def read_file(self):

        logger.info("Reading CONLL (train) file")

        data = []
        head2mod = []
        datum = self.init_datum()
        depG = nx.DiGraph()

        self.vocab = ["<unk>", "<null>"]
        PAS = self.extract_semroles()

        pas_idx = 0
        curr_pas = PAS[pas_idx]

        with open(self.fname, "r") as conll_result:
            reader = csv.reader(conll_result, delimiter="\t")

            for row in reader:
                if row[0].isdigit():

                    # process one line
                    self.read_row(datum, row, depG, head2mod)

                else:
                    # EOS
                    self.process_newline(datum, depG, head2mod, data, curr_pas)

                    # init new datum
                    head2mod = []
                    datum = self.init_datum()
                    depG = nx.DiGraph()
                    pas_idx += 1

                    try:
                        curr_pas = PAS[pas_idx]
                    except IndexError:
                        break

        paslen = len(PAS)
        datalen = len(data)
        assert paslen == datalen, "number of pas %d != number of data %d" % (paslen, datalen)

        return data, self.vocab


class ConllTestFileReader(ConllFileReader):

    def read_row(self, curr_datum, row, dG, h2m):

        position = int(row[0]) - 1
        head_position = int(row[5]) - 1

        word = row[1]
        lem = row[2]
        pos = row[3]
        ne = row[4]
        deprel = row[6]

        if row[11] == "Y":
            items = row[12].split(".")
            if len(items) == 2:
                sense = int(items[1])
                ispred = True

            else:
                # this happens on urls
                logger.info("Strange predicate: %s" % row[12])
                ispred = False  # fixme
                sense = None
                pass

        else:
            ispred = False
            sense = None

        # TODO makes sense if NN is used
        # token = {"word": self.vocab.get(word, "<unk>"), "lemma": self.vocab.get(lem, "<unk>"), "pos": self.vocab.get(pos, "<unk>"),
        #          "ne": self.vocab.get(ne, "<unk>"), "rel": self.vocab.get(deprel, "<unk>"),
        #          "ispred": ispred, "sense": self.vocab.get(sense, "<unk>"), "preds": None, "args": None,
        #          "head": head_position, "lsb": None, "rsb": None,
        #          "netype": None, "ctype": None, "concepts": [None], "neg": False, "node_id": [], "ci_data": (),
        #
        #          }

        token = {"word": word, "lemma": lem, "pos": pos, "ne": ne, "rel": deprel,
                 "ispred": ispred, "sense": sense, "preds": None, "args": None,
                 "head": head_position, "lsb": None, "rsb": None,
                 "netype": None, "ctype": None, "concepts": [None], "neg": False, "node_id": [], "ci_data": (),

                 }

        dG.add_node(position, attr_dict=token)

        if deprel != "ROOT":
            # (parent, child, role) in the dependency tree
            h2m.append((head_position, position, {"label": token["rel"]}))
        else:
            dG.graph["root"] = int(row[0]) - 1

        curr_datum["tokens"].append(token)

    def extract_semroles(self):

        with open(self.fname, "r") as conll_result:

            reader = csv.reader(conll_result, delimiter="\t")

            all_pas = []

            pred_idx = collections.defaultdict()
            pred_counter = 0
            pas_triples = []

            for row in reader:
                if row[0].isdigit():
                    tokid = int(row[0]) - 1

                    if row[11] == "Y":
                        pred_idx[pred_counter] = tokid
                        pred_counter += 1

                    for col_idx, col_data in enumerate(row[13:]):
                        if col_data != "_":
                            pas_triples.append((tokid, col_idx, col_data))

                else:
                    # EOS
                    preds = collections.defaultdict(rec_dd)
                    args = collections.defaultdict(rec_dd)

                    for triple in pas_triples:
                        tokid, pred_num, role = triple

                        # TODO for NN
                        # role = self.vocab.get(role, "<unk>")

                        pred_tokid = pred_idx[pred_num]
                        preds.setdefault(pred_tokid, []).append((tokid, role))
                        args.setdefault(tokid, []).append((pred_tokid, role))

                    all_pas.append((preds, args))

                    pred_idx = {}
                    pred_counter = 0
                    pas_triples = []

            return all_pas

    def read_file(self):

        logger.info("Reading CONLL (test) file")

        # TODO need to change this if use NN
        assert type(self.vocab) == list

        data = []
        head2mod = []
        datum = self.init_datum()
        depG = nx.DiGraph()
        PAS = self.extract_semroles()

        pas_idx = 0
        curr_pas = PAS[pas_idx]

        with open(self.fname, "r") as conll_result:
            reader = csv.reader(conll_result, delimiter="\t")

            for row in reader:
                if row[0].isdigit():

                    # process one line
                    self.read_row(datum, row, depG, head2mod)

                else:
                    # EOS
                    self.process_newline(datum, depG, head2mod, data, curr_pas)

                    # init new datum
                    head2mod = []
                    datum = self.init_datum()
                    depG = nx.DiGraph()
                    pas_idx += 1

                    try:
                        curr_pas = PAS[pas_idx]
                    except IndexError:
                        break

        paslen = len(PAS)
        datalen = len(data)
        assert paslen == datalen, "number of pas %d != number of data %d" %(paslen, datalen)

        return data