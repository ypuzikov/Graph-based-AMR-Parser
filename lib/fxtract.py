#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, collections
import logging
import numpy as np
import time
import networkx as nx
import itertools
import string
from sklearn.feature_extraction import DictVectorizer


logger = logging.getLogger("main")

BOS_token = dict(zip(["word", "lemma", "pos", "ne", "rel", "ctype"], ["<bos>"] * 5 + ["empty"]))
EOS_token = dict(zip(["word", "lemma", "pos", "ne", "rel", "ctype"], ["<eos>"] * 5 + ["empty"]))
ROOT_token = dict(zip(["word", "lemma", "pos", "ne", "rel", "ctype"], ["<root>"] * 5 + ["empty"]))

def iter_corpus(corpus):
    for instance in corpus:
        yield instance

def rec_dd():
    return collections.defaultdict()


def gen_labels(labelset):
    for l in labelset:
        yield l


# # http://stackoverflow.com/questions/22118350/python-sentiment-analysis-using-pointwise-mutual-information
def pmi(word1, word2, unigram_freq, bigram_freq):
    prob_word1 = unigram_freq[word1] / float(sum(unigram_freq.values()))
    prob_word2 = unigram_freq[word2] / float(sum(unigram_freq.values()))
    prob_word1_word2 = bigram_freq[" ".join([word1, word2])] / float(sum(bigram_freq.values()))
    if prob_word1 * prob_word2 != 0 and prob_word1_word2 != 0:
        return np.log(prob_word1_word2 / float(prob_word1 * prob_word2), 2)
    else:
        return -1e30


class Fxtractor(object):
    def __init__(self, fx_maps=None, labels=None, flist=None):
        self.fx_maps = fx_maps if not None else collections.defaultdict()
        self.labels = labels
        self.flist = flist
        self.stats = collections.defaultdict()

    # CI =========================================================

    def tok_fdict_generator(self, instance):
        tokens = instance["tokens"]
        depG = instance["depG"]
        for tokid, tok in enumerate(tokens):
            ctype = tok["ctype"]
            if ctype != "misc":
                yield (tokid, self.fdict_token(tokens,depG, tokid), ctype)

    def fill_lsb_feats(self, tok, features, tokens):

        lsbs = tok["lsb"]
        if len(lsbs) > 1:
            lsb1, lsb2 = lsbs[:2]
        elif len(lsbs) > 0:
            lsb1, lsb2 = lsbs[0], None
        else:
            lsb1, lsb2 = None, None

        if lsb1 is not None:
            features["i_lsb1_w"] = (tokens[lsb1]["word"], 1)
            features["i_lsb1_p"] = (tokens[lsb1]["pos"], 1)
        else:
            features["i_lsb1_w"] = ("<null>", 1)
            features["i_lsb1_p"] = ("<nul>", 1)

        if lsb2 is not None:
            features["i_lsb2_w"] = (tokens[lsb2]["word"], 1)
            features["i_lsb2_p"] = (tokens[lsb2]["pos"], 1)
        else:
            features["i_lsb2_w"] = ("<null>", 1)
            features["i_lsb2_p"] = ("<null>", 1)

    def fill_rsb_feats(self, tok, features, tokens):

        rsbs = tok["rsb"]
        if len(rsbs) > 1:
            rsb1, rsb2 = rsbs[:2]
        elif len(rsbs) > 0:
            rsb1, rsb2 = rsbs[0], None
        else:
            rsb1, rsb2 = None, None

        if rsb1 is not None:
            features["i_rsb1_w"] = (tokens[rsb1]["word"], 1)
            features["i_rsb1_p"] = (tokens[rsb1]["pos"], 1)
        else:
            features["i_rsb1_w"] = ("<null>", 1)
            features["i_rsb1_p"] = ("<null>", 1)

        if rsb2 is not None:
            features["i_rsb2_w"] = (tokens[rsb2]["word"], 1)
            features["i_rsb2_p"] = (tokens[rsb2]["pos"], 1)
        else:
            features["i_rsb2_w"] = ("<null>", 1)
            features["i_rsb2_p"] = ("<null>", 1)

    def fdict_token(self, tokens, dG, tokid):

        features = {}
        selected_features = collections.defaultdict()

        # --------------------------------------------------------- left context

        prev3 = tokens[tokid - 3] if tokid - 3 >= 0 else BOS_token
        p3w = prev3["word"]
        p3p = prev3["pos"]

        prev2 = tokens[tokid - 2] if tokid - 2 >= 0 else BOS_token
        p2w = prev2["word"]
        p2p = prev2["pos"]

        prev1 = tokens[tokid - 1] if tokid - 1 >= 0 else BOS_token
        p1w = prev1["word"]
        p1p = prev1["pos"]
        p1d = prev1["rel"]

        # i - 3
        features["i-3_w"] = (p3w, 1)
        features["i-3_p"] = (p3p, 1)

        # i - 2
        features["i-2_w"] = (p2w, 1)
        features["i-2_p"] = (p2p, 1)

        # i - 1
        features["i-1_w"] = (p1w, 1)
        features["i-1_p"] = (p1p, 1)

        # ----------------------------------------------------------- center word

        center = tokens[tokid]
        cw = center["word"]
        cl = center["lemma"]
        cp = center["pos"]
        cn = center["ne"]
        cd = center["rel"]

        features["i_w_1symbol-quote"] = ((cw[0] == '"'), 1)
        features["i_w_1symbol-cap"] = ((cw[0].isupper()), 1)
        features["i_w-2symbol-cap"] = ((len(cw) > 2 and cw[1].isupper()), 1)
        features["i_w-all-digit"] = ((cw.isdigit()), 1)
        features["i_w_lastsymbol-letter"] = ((cw[-1].isalpha()), 1)
        features["i_w_contains-dot"] = (("." in cw), 1)
        features["i_w_contains-hyphen"] = (("-" in cw), 1)

        features["i_l-ly_adv"] = ((cl[-2:] == "ly" and cp == "RB"), 1)
        features["i_w"] = ((cw), 1)
        features["i_p"] = ((cp), 1)
        features["i_ne"] = ((cn), 1)
        features["i_w_pref"] = ((cw[0:3]), 1)
        features["i_w_suf"] = ((cw[-3:]), 1)
        # add("i_ispred", center["ispred"])

        # ----------------------------------------------------------- right context

        next1 = tokens[tokid + 1] if tokid + 1 <= 0 else EOS_token
        n1w = next1["word"]
        n1p = next1["pos"]
        n1d = next1["rel"]

        next2 = tokens[tokid + 2] if tokid + 2 <= 0 else EOS_token
        n2w = next2["word"]
        n2p = next2["pos"]

        next3 = tokens[tokid + 3] if tokid + 3 <= 0 else EOS_token
        n3w = next3["word"]
        n3p = next3["pos"]

        # i + 1
        features["i+1_w"] = (n1w, 1)
        features["i+1_p"] = (n1p, 1)

        # i + 2
        features["i+2_w"] = (n2w, 1)
        features["i+2_p"] = (n2p, 1)

        # -----------------------------------------------------------  tree context

        # # out_arcs = [e[2]["label"] for e in filter(lambda e: e[0]==tokid, dG.edges(data=True))]
        # # TODO rsb2 and rsb1 give only 0.04 improvement
        # try:
        #     par1id = dG.predecessors(tokid)[0]
        #     par1tok = tokens[par1id]
        # except IndexError:
        #     par1tok = ROOT_token
        #
        # par1w = par1tok["word"]
        # par1p = par1tok["pos"]
        # par1d = par1tok["rel"]

        # -------------------------------------------------------  BIGRAM, TRIGRAM

        features["i-1_rel+i_rel"] = ("+".join([p1d, cd]), 1)
        features["i_rel+i+1_rel"] = ("+".join([cd, n1d]), 1)

        features["i_w+i+1_w"] = ("+".join([cw, n1w]), 1)
        features["i-1_p+i_p+i+1_p"] = ("+".join([p1p, cp, n1p]), 1)

        # ====================================================================
        # # experimental
        # features["par1_w"] = (par1w, 1)
        # features["par1_p"] = (par1p, 1)
        # features["par1_rel"] = (par1d, 1)

        # # TODO: works, but need tweaking the ctypes for Nones
        # if p1w[0].isdigit():
        #     p1ct = "digit"
        #
        # elif p1w[0] in string.punctuation:
        #     p1ct = "punct"
        #
        # elif prev1["ctype"] is None:
        #     # print "ololo", p1w
        #     p1ct = "empty"
        #
        # else:
        #     p1ct = prev1["ctype"]
        #
        # if p2w[0].isdigit():
        #     p2ct = "digit"
        #
        # elif p2w[0] in string.punctuation:
        #     p2ct = "punct"
        #
        # elif prev2["ctype"] is None:
        #     # print "ololo", p1w
        #     p2ct = "empty"
        #
        # else:
        #     p2ct = prev2["ctype"]


        # features["i-2_ctype+i-1_ctype"] = ("+".join([p2ct, p1ct]), 1)

        # bad
        # features["i-1_w+i_w+i-1_rel"] = ("+".join([p1w, cw, p1d]), 1)
        # features["i_w+i_p+i_ne"] = ("+".join([cw, cp, cn]), 1)
        # features["i-2_pos+i-1_pos+i_pos"] = ("+".join([p2p, p1p, cp]), 1)
        # features["i_pos+i+1_pos+i+2_pos"] = ("+".join([cp, n1p, n2p]), 1)
        # features["i_w_len"] = ((len(cw)), 1)
        # features["i_out_deps"] = (("+".join(out_arcs)), 1)
        # features["i_num_out_deps"] = ((len(out_arcs)), 1)
        # features["i_ispron"] = ((cp.startswith("P")), 1)
        # features["i_ispron"] = ((cp == "PRP"), 1)

        for fk in self.flist:
            fv = features[fk]
            selected_features["%s=%s" % (fk, fv[0])] = fv[1]

        return selected_features

    def fx_dataset_ci(self, dataset, train=False):

        start = time.clock()

        if train:

            logger.info("Fitting vectorizers ...")            
            ci_l2i = dict(zip(self.labels, range(len(self.labels))))
            ci_i2l = dict(zip(ci_l2i.values(), ci_l2i.keys()))

            ci_xvec = DictVectorizer()
            ci_xvec.fit([x[1] for instance in iter_corpus(dataset)
                         for x in self.tok_fdict_generator(instance)
                         ])

            self.fx_maps["ci_l2i"] = ci_l2i
            self.fx_maps["ci_i2l"] = ci_i2l
            self.fx_maps["ci_xmap"] = ci_xvec
            self.fx_maps["ci_fdim"] = len(ci_xvec.get_feature_names())

        else:
            ci_l2i = self.fx_maps["ci_l2i"]
            ci_xvec = self.fx_maps["ci_xmap"]

        logger.info("Extracting features ...")

        for instance in iter_corpus(dataset):
            for token_data in self.tok_fdict_generator(instance):
                tokid, feats, gctype = token_data
                try:
                    instance["tokens"][tokid]["ci_data"] = (ci_xvec.transform(feats), ci_l2i[gctype])
                except KeyError:
                    # print "Unknown concept type: %s" % gctype
                    instance["tokens"][tokid]["ci_data"] = (ci_xvec.transform(feats), -1)

        logger.info("Feature extraction took %0.2f seconds" % (time.clock() - start))

    # RI =========================================================

    # # good - to add last
    # def hc_tc_CP(feats, head_concept, tail_concept)
    #     if "%s+%s" % (head_concept, tail_concept) in self.fx_maps["edge_conc_lh"].conditions():
    #         for e in self.fx_maps["edges"]:
    #             feats["hc+tc+%s_CP" % e] = self.fx_maps["edge_conc_lh"]["%s+%s" % (head_concept, tail_concept)].prob(e)



    def get_spath_all(self, depG):
        """
        Compute shortest paths between each pair of nodes in the graph.
        Returns a dictionary of the form {a: {a:[a], b:[a,c,b], ...}}
        Note: includes a self-path.

        """

        allpaths = {}
        for node, node_ats in depG.nodes(data=True):
            allpaths[node] = nx.shortest_path(depG, source=node)

        return allpaths

    def get_spath_pair(self, span1, span2, all_paths):
        shortest_len = np.inf
        best_path = None
        # best_node_pair = None

        # print "Start span: %s, End span: %s" %(span1, span2)
        for start in range(*span1):
            for end in range(*span2):
                # print "start %d, end %d" %(start, end)
                if end in all_paths[start]:
                    path = all_paths[start][end]
                    length = len(path)
                    if length < shortest_len:
                        best_path = path
                        shortest_len = length
                        # print "Path: %s, length: %d" %(path, length)
                        # best_node_pair = (start,end)

        return shortest_len, best_path



    def get_path_feats(self, path_length, path, dep_graph):

        try:
            pos_path_list = ["%s" % (dep_graph.node[path[0]]["pos"])]
            dep_path_list = []

            for token_id in path[1:]:
                pos_path_list.append(dep_graph.node[token_id]["pos"])
                dep_path_list.append(dep_graph.node[token_id]["rel"])

            pos_path_str = "_".join(pos_path_list)
            dep_path_str = "_".join(dep_path_list)

        except TypeError:
            pos_path_str = False
            dep_path_str = False

        return pos_path_str, dep_path_str

    def get_token_feats(self, span_start, span_end, tokens):
        toks = [tokens[span_start]] if span_start == span_end else tokens[span_start:span_end]

        words = []
        pos = []
        lemmas = []
        nes = []
        ctypes = []
        srls = False

        for T in toks:
            words.append(T["word"])
            pos.append(T["pos"])
            lemmas.append(T["lemma"])
            nes.append(T["ne"])
            ctypes.append(T["ctype"])

            if T["ispred"]:
                srls = True

        wstr = "+".join(words)
        lemstr = "+".join(lemmas)
        pstr = "+".join(pos)
        nestr = "+".join(nes)
        ctypestr = "+".join([c if c is not None else "empty" for c in ctypes])


        return wstr, lemstr, pstr, nestr, ctypestr, srls

    def get_dist_feat(self, hspan, hspan_start, hspan_end, tspan, tspan_start, tspan_end):

        if hspan == tspan:
            dist = 0.0
        elif hspan_start >= tspan_end:
            dist = (hspan_start - tspan_end + 1)
        else:
            dist = (tspan_start - hspan_end + 1)

        return dist

    def get_dephead_token(self, nid, aG, toposort):
        start, end = aG.node[nid]["span"]
        tokid_range = end - start
        if tokid_range == 1:
            head_tokid = start
            head_idx = toposort[head_tokid]

        else:
            tokids = range(start, end)
            head_idx = min([toposort[tokid] for tokid in tokids])
            head_tokid = [tokid for tokid in tokids if toposort[tokid] == head_idx][0]

        return (head_idx, head_tokid)

    def fdict_edge(self, tokens, dep_graph, synpaths, amr_graph, hid, tid, toposort):

        """ Edge dictionary feature extraction (RI stage) """

        feats = {}
        selected_features = {}

        head_concept = amr_graph.node[hid]["concept"]
        tail_concept = amr_graph.node[tid]["concept"]
        hspan = amr_graph.node[hid]["span"]
        tspan = amr_graph.node[tid]["span"]
        hspan_start, hspan_end = hspan
        tspan_start, tspan_end = tspan

        # populating the feats dictionary
        feats["bias"] = (1, 1)
        feats["hc"] = (head_concept, 1)
        feats["tc"] = (tail_concept, 1)

        feats["hc1q"] = (head_concept[0] == '"', 1)
        feats["hcd"] = (head_concept.isdigit(), 1)
        feats["hch"] = ("-" in head_concept, 1)
        feats["hc1c"] = (head_concept[0].isupper(), 1)
        feats["hc2c"] = (len(head_concept) > 2 and head_concept[1].isupper(), 1)
        feats["hcll"] = (head_concept[-1].isalpha(), 1)

        feats["tc1q"] = (tail_concept[0] == '"', 1)
        feats["tcd"] = (tail_concept.isdigit(), 1)
        feats["tch"] = ("-" in tail_concept, 1)
        feats["tc1c"] = (tail_concept[0].isupper(), 1)
        feats["tc2c"] = (len(tail_concept) > 2 and tail_concept[1].isupper(), 1)
        feats["tcll"] = (tail_concept[-1].isalpha(), 1)

        # path features
        path_length, path = self.get_spath_pair(hspan, tspan, synpaths)
        pos_path_str, dep_path_str = self.get_path_feats(path_length, path, dep_graph)

        # token features
        hwstr, hlstr, hpstr, hnestr, hctypestr, hsrl_bool = self.get_token_feats(hspan_start, hspan_end, tokens)
        twstr, tlstr, tpstr, tnestr, tctypestr, tsrl_bool = self.get_token_feats(tspan_start, tspan_end, tokens)

        # distance feature
        dist = self.get_dist_feat(hspan, hspan_start, hspan_end, tspan, tspan_start, tspan_end)

        # feats["hn"] = (hnestr, 1)
        # feats["tn"] = (tnestr, 1)
        # feats["hn+tn"] = ("+".join((hnestr, tnestr)), 1)

        feats["hw"] = (hwstr, 1)
        feats["tw"] = (twstr, 1)
        feats["hp"] = (hpstr, 1)
        feats["tp"] = (tpstr, 1)
        feats["hp_tp"] = ("+".join((hpstr, tpstr)), 1)
        feats["ppath"] = (pos_path_str, 1)
        feats["dpath"] = (dep_path_str, 1)
        feats["path_len"] = (path_length, 1)
        feats["dist"] = (dist, dist)

        hid_headword_num, hid_headid = self.get_dephead_token(hid, amr_graph,toposort)
        tid_headword_num, tid_headid = self.get_dephead_token(tid, amr_graph, toposort)

        hid_headword_word = tokens[hid_headid]["word"]
        tid_headword_word = tokens[tid_headid]["word"]

        hid_headword_pos = tokens[hid_headid]["pos"]
        tid_headword_pos = tokens[tid_headid]["pos"]

        hid_headword_rel = tokens[hid_headid]["rel"]
        tid_headword_rel = tokens[tid_headid]["rel"]

        feats["h_headw"] = (hid_headword_word, 1)
        feats["t_headw"] = (tid_headword_word, 1)
        feats["h_headw+t_headw"] = ("+".join((hid_headword_word, tid_headword_word)), 1)

        feats["h_headp"] = (hid_headword_pos, 1)
        feats["t_headp"] = (tid_headword_pos, 1)
        feats["h_headp+t_headp"] = ("+".join((hid_headword_pos, tid_headword_pos)), 1)

        feats["h_headd"] = (hid_headword_rel, 1)
        feats["t_headd"] = (tid_headword_rel, 1)
        feats["h_headd+t_headd"] = ("+".join((hid_headword_rel, tid_headword_rel)), 1)

        # feats["hw_tw"] = ("+".join((hwstr, twstr)), 1)
        # feats["hsrl"] = (hsrl_bool, 1)
        # feats["tsrl"] = (tsrl_bool, 1)

        # hid_head_word_ispred = tokens[hid_headid]["ispred"]
        # tid_head_word_ispred = tokens[tid_headid]["ispred"]
        # feats["h_head_ispred"] = (hid_head_word_ispred, 1)
        # feats["t_head_ispred"] = (tid_head_word_ispred, 1)

        hid_headword_preds = tokens[hid_headid]["preds"]
        hid_headword_args = tokens[hid_headid]["args"]

        if hid_headword_args is not None:

            args_roles = {}
            for a, r in hid_headword_args:
                args_roles[a] = r

            if tid_headid in args_roles:
                feats["t_head_in_h_head_args"] = (True, 1)
                feats["h_head_srl_role"] = (args_roles[tid_headid], 1)
            else:
                feats["t_head_in_h_head_args"] = (False, 1)
                feats["h_head_srl_role"] = (False, 1)
        else:

            feats["t_head_in_h_head_args"] = (False, 1)
            feats["h_head_srl_role"] = (False, 1)



        # # path to root
        # deproot = dep_graph.graph["root"]
        # h_head_path = synpaths[deproot][hw_headid]
        # t_head_path = synpaths[deproot][tw_headid]

        # print " ".join([T["word"] for T in tokens])
        # print hw_word, tw_word
        # h_head_path_word = " ".join([tokens[T]["word"] for T in h_head_path])
        # t_head_path_word = " ".join([tokens[T]["word"] for T in t_head_path])

        # t_head_path_len, t_head_path = self.get_spath_pair(tspan, (deproot, deproot + 1), synpaths)
        # t_head_path_pos_str, t_head_path_dep_str = self.get_path_feats(t_head_path_len, t_head_path, dep_graph)



        # feats["hppath"] = (h_pos_path_to_deproot_str, 1)
        # feats["hdpath"] = (h_dep_path_to_deproot_str, 1)
        # feats["hppath+hdpath"] = ("+".join([str(h_pos_path_to_deproot_str), str(h_dep_path_to_deproot_str)]), 1)
        #
        # feats["tppath"] = (t_pos_path_to_deproot_str, 1)
        # feats["tdpath"] = (t_dep_path_to_deproot_str, 1)
        # feats["tppath+tdpath"] = ("+".join([str(t_pos_path_to_deproot_str), str(t_dep_path_to_deproot_str)]), 1)
        #
        # feats["hppath+tppath"] = ("+".join([str(h_pos_path_to_deproot_str), str(t_pos_path_to_deproot_str)]), 1)
        # feats["hdpath+tdpath"] = ("+".join([str(h_dep_path_to_deproot_str), str(t_dep_path_to_deproot_str)]), 1)
        # feats["hppath+hdpath+tppath+tdpath"] = ("+".join([str(h_pos_path_to_deproot_str),
        #                                                   str(h_dep_path_to_deproot_str),
        #                                                   str(t_pos_path_to_deproot_str),
        #                                                   str(t_dep_path_to_deproot_str)]), 1)


        # feats["hclen"] = (len(head_concept), len(head_concept))
        # feats["tclen"] = (len(tail_concept), len(tail_concept))

        # computing pmi values
        # hc_tc_pmi = pmi(head_concept, tail_concept,self.fx_maps["unifreq"], self.fx_maps["bifreq"])

        # # fixme finding the root tokens of each span is not useful - why?
        # if hspan_start == hspan_end:  # this happens in the training data only if concept is "-"
        #     h_toposort_seq = [(hspan_start, toposort[hspan_start])]
        # else:
        #     h_toposort_seq = [(x, toposort[x]) for x in range(hspan_start, hspan_end)]
        #
        # if tspan_start == tspan_end:
        #     t_toposort_seq = [(tspan_start, toposort[tspan_start])]
        # else:
        #     t_toposort_seq = [(x, toposort[x]) for x in range(tspan_start, tspan_end)]
        #
        # h_root = min(h_toposort_seq, key=operator.itemgetter(1))[0]
        # t_root = min(t_toposort_seq, key=operator.itemgetter(1))[0]

        # parents
        head_parents = [nid for nid in amr_graph.predecessors(hid) if amr_graph.node[nid]["span"][0] is not None]
        tail_parents = [nid for nid in amr_graph.predecessors(tid) if amr_graph.node[nid]["span"][0] is not None]
        num_head_parents = len(head_parents)
        num_tail_parents = len(tail_parents)

        head_sbls = []
        for pid in head_parents:
            for chid in amr_graph.successors(pid):
                # if chid not in [hid, tid]:
                head_sbls.append(chid)

        tail_sbls = []
        for pid in tail_parents:
            for chid in amr_graph.successors(pid):
                # if chid not in [hid, tid]:
                tail_sbls.append(chid)

        num_head_sbls = len(head_sbls)
        num_tail_sbls = len(tail_sbls)

        # -------------------------------------------------------- LSB , RSB

        # # LSB
        # lsbs = center["lsb"]
        # if len(lsbs) > 1:
        #     lsb1, lsb2 = lsbs[:2]
        # elif len(lsbs) > 0:
        #     lsb1, lsb2 = lsbs[0], None
        # else:
        #     lsb1, lsb2 = None, None
        #
        # if lsb1 is not None:
        #     add("i_lsb1_w", tokens[lsb1]["word"])
        #     add("i_lsb1_pos", tokens[lsb1]["pos"])
        #
        # if lsb2 is not None:
        #     add("i_lsb2_w", tokens[lsb2]["word"])
        #     add("i_lsb2_pos", tokens[lsb2]["pos"])

        # # RSB

        # rsbs = center["rsb"]
        # if len(rsbs) > 1:
        #     rsb1, rsb2 = rsbs[:2]
        # elif len(rsbs) > 0:
        #     rsb1, rsb2 = rsbs[0], None
        # else:
        #     rsb1, rsb2 = None, None
        #
        # if rsb1 is not None:
        #     add("i_rsb1_w", tokens[rsb1]["word"])
        #     add("i_rsb1_pos", tokens[rsb1]["pos"])
        #
        # if rsb2 is not None:
        #     add("i_rsb2_w", tokens[rsb2]["word"])
        #     add("i_rsb2_pos", tokens[rsb2]["pos"])

        # head_next_tok1 = tokens[hspan_end] if hspan_end <= len(tokens) -1 else EOS_token
        # head_next_tok1_w = head_next_tok1["word"]
        # head_next_tok1_p = head_next_tok1["pos"]
        #
        # feats["hn1w"] = (head_next_tok1_w, 1)
        # feats["hn1p"] = (head_next_tok1_p, 1)

        # fixme these feats slightly harm when used with the hw, tw, hp, tp
        # feats["hheadw"] = (tokens[h_root]["word"],1)
        # feats["theadw"] = (tokens[t_root]["word"],1)
        # feats["hheadp"] = (tokens[h_root]["pos"],1)
        # feats["theadp"] = (tokens[t_root]["pos"],1)

        # feats["hc_verb"] = (re.match(r"\w+-.*\d+",head_concept),1)
        # feats["path_length"] = (path_length,1)
        # feats["hc_tc_pmi"] = (hc_tc_pmi, hc_tc_pmi)

        # feats["dpath_hctype_tctype"] = (dep_path_str, hpstr, tpstr)
        # feats["hp_tp"] = (hpstr, tpstr)

        # if "%s+%s" % (head_concept, tail_concept) in self.fx_maps["edge_conc_lh"].conditions():
        #     for e in self.fx_maps["edges"]:
        #         feats["hc+tc+%s_CP" % e] = self.fx_maps["edge_conc_lh"]["%s+%s" % (head_concept, tail_concept)].prob(e)

        # feats["hp_tp_dpath=%s+%s+%s" % (hpstr, tpstr, dep_path_str)] = 1
        # if head_concept in self.fx_maps["edge_hconc_lh"].conditions():
        #     for e in self.fx_maps["edges"]:
        #         feats["hc=%s+e=%s_CP" %(head_concept,e)] = self.fx_maps["edge_hconc_lh"][head_concept].prob(e)
        #         # print "hc=%s+e=%s_CP" %(head_concept,e),feats["hc=%s+e=%s_CP" %(head_concept,e)]


        # feats["hw"] = (hwstr, 1)
        # feats["head_isppred"] = (, 1)

        for fk in self.flist:
            fv = feats[fk]
            selected_features["%s=%s" % (fk, fv[0])] = fv[1]

        return selected_features

    def edge_fdict_generator(self, instance):

        tokens = instance["tokens"]
        gG = instance["amrG"]
        depG = instance["depG"]
        syn_paths = instance["spaths"]
        toposort = dict([(nid, idx) for idx, nid in enumerate(nx.topological_sort(depG, reverse=False))])

        # not extracting features for unaligned nodes
        aligned_nodes = filter(lambda x: gG.node[x]["span"][0] is not None, gG.nodes())

        for node_pair in itertools.combinations(aligned_nodes, 2):
            hid, tid = node_pair

            hcl = gG[hid][tid]["label"] if (hid in gG and tid in gG[hid]) else "<null>"
            chl = gG[tid][hid]["label"] if (tid in gG and hid in gG[tid]) else "<null>"

            hcf = self.fdict_edge(tokens, depG, syn_paths, gG, hid, tid, toposort)
            chf = self.fdict_edge(tokens, depG, syn_paths, gG, tid, hid, toposort)
            yield (hid, tid, hcf, hcl, chf, chl)

    def fx_dataset_ri(self, dataset, train=False):

        start = time.clock()

        if train:

            # TRAINING DATA => extract vocab
            logger.info("Fitting vectorizers ...")
            ri_l2i = dict(zip(self.labels, range(len(self.labels))))
            ri_i2l = {}

            for k, v in ri_l2i.items():
                ri_i2l[v] = k

            ri_xvec = DictVectorizer()
            ri_xvec.fit([x for instance in iter_corpus(dataset)
                         for node_data in self.edge_fdict_generator(instance)
                         for x in [node_data[2], node_data[4]]
                         ])

            self.fx_maps["ri_l2i"] = ri_l2i
            self.fx_maps["ri_i2l"] = ri_i2l
            self.fx_maps["ri_xmap"] = ri_xvec
            self.fx_maps["ri_fdim"] = len(ri_xvec.get_feature_names())

        else:

            # TEST MODE
            ri_l2i = self.fx_maps["ri_l2i"]
            ri_xvec = self.fx_maps["ri_xmap"]

        logger.info("Extracting features ...")

        for instance in iter_corpus(dataset):

            for node_data in self.edge_fdict_generator(instance):
                hid, tid, hcf, hcl, chf, chl = node_data
                h_feat = ri_xvec.transform(hcf)
                ch_feat = ri_xvec.transform(chf)

                try:
                    h_lab = ri_l2i[hcl]
                except KeyError:
                    h_lab = ri_l2i["<unk>"]

                try:
                    ch_lab = ri_l2i[chl]
                except KeyError:
                    ch_lab = ri_l2i["<unk>"]

                instance["ri_data"][hid][tid] = (h_feat, h_lab)
                instance["ri_data"][tid][hid] = (ch_feat, ch_lab)

        assert self.fx_maps["ri_i2l"][0] == "<null>" and self.fx_maps["ri_l2i"]["<null>"] == 0
        logger.info("Feature extraction took %0.2f seconds" % (time.clock() - start))