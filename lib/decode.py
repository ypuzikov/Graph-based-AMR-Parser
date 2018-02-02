#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import pprint
import networkx as nx
import operator
import numpy as np
import itertools
import collections
import logging
import time

logger = logging.getLogger("main")

class Decoder(object):

    def __init__(self, arg_elabs, arg_ids, stepsize = 0.001, maxiter = 1000):

        # ====================== LAGRANGIAN RELAXATION PARAMETERS ==========================

        self.lr_stepsize = stepsize
        self.lr_maxiter = maxiter

        self.arg_labs = arg_elabs
        self.arg_ids = arg_ids

        self.mu = self.b = self.A = None
        self.mu2id = collections.defaultdict()
        self.id2mu = collections.defaultdict()

        # count non-deterministic AMRs which LR failed to recover
        self.failed = 0
        self.success = 0

    def init_multipliers(self, G):
        idx = 0
        for nid in G.nodes():
            for role_id in self.arg_ids:
                self.mu2id[(nid, role_id)] = idx
                self.id2mu[idx] = (nid, role_id)  # fixme not needed!
                idx += 1

        self.b = np.ones(idx)
        self.mu = np.zeros(idx)
        self.A = np.zeros(idx)

    def MSCG(self, G, E_neg):

        Q = sorted(E_neg, key=operator.itemgetter(1), reverse=True)
        num_weakly_connected = nx.number_weakly_connected_components(G)

        while len(Q) > 0 and not nx.is_weakly_connected(G):

            try:
                candidate_edge, candidate_score = Q.pop(0)
            except IndexError:
                break

            head, tail, label = candidate_edge
            if tail in G[head] or head in G[tail]:
                continue

            G.add_edge(*candidate_edge)
            new_num_weakly_connected = nx.number_weakly_connected_components(G)
            if new_num_weakly_connected < num_weakly_connected:
                num_weakly_connected = new_num_weakly_connected
            else:
                G.remove_edge(*candidate_edge[:2])

        assert nx.is_weakly_connected(G)

    def decode_one(self, feats, G, scorer, fmaps):

        E_neg = []  # E_neg ($e \in E | \theta^{T} g(e) \leq 0$) - prioritize by scores
        heads, non_vars = self.group_nodes(G)

        # print "Heads:", " ".join(na["concept"] for nid, na in G.nodes(data=True) if na["var"] is not None)
        # print "Tails:", " ".join(na["concept"] for nid, na in G.nodes(data=True) if na["var"] is None)

        for tail in non_vars:
            if len(G.predecessors(tail))>0:
                continue

            elif len(heads) == 0:
                for nid in non_vars:
                    heads.append(nid)

                break
            else:
                self.decode_1d(heads, tail, feats, G, scorer, fmaps)

        # assign edges between all nodes
        for node_pair in itertools.combinations(heads, 2):
            nid1, nid2 = node_pair

            if nid1 in G[nid2] or nid2 in G[nid1]:
                continue

            # -----------------------------------------------
            # scoring 1 direction

            x1, y1 = feats[nid1][nid2]
            x_ind, x_data = x1.indices, x1.data
            scores1 = scorer(x_ind, x_data)
            sorted_labids1 = np.argsort(scores1, axis=None)
            l12, l11 = sorted_labids1[-2:]
            s11, s12 = scores1[l11], scores1[l12]

            e11 = (nid1, nid2, {"label": fmaps["ri_i2l"][l11]})
            e12 = (nid1, nid2, {"label": fmaps["ri_i2l"][l12]})

            # -----------------------------------------------
            # scoring 2 direction

            x2, y2 = feats[nid2][nid1]
            x_ind, x_data = x2.indices, x2.data
            scores2 = scorer(x_ind, x_data)
            sorted_labids2 = np.argsort(scores2, axis=None)
            l22, l21 = sorted_labids2[-2:]
            s21, s22 = scores2[l21], scores2[l22]

            e21 = (nid2, nid1, {"label": fmaps["ri_i2l"][l21]})
            e22 = (nid2, nid1, {"label": fmaps["ri_i2l"][l22]})
            # -----------------------------------------------

            if l11 == 0:
                E_neg.append((e12, s12))
                if l21 == 0:
                    E_neg.append((e22, s22))
                else:
                    G.add_edge(*e21)

            elif l21 != 0:
                if s11 > s21:
                    G.add_edge(*e11)
                else:
                    G.add_edge(*e21)
            else:
                G.add_edge(*e11)

        if len(E_neg) == 0:
            return

        self.MSCG(G, E_neg)
        return
        # if not self.check_determinism(G):
        #     # logger.info("G is not deterministic! Decoding with Lagrangian Relaxation")
        #     self.init_multipliers(G)
        #     self.decode_constrained(G, feats, scorer, fmaps)

    def check_determinism(self, G):

        all_edges = G.edges(data=True)
        arg_edges = filter(lambda x: x[2]["label"] in self.arg_labs, all_edges)

        if len(arg_edges) == 0:
            return True

        heads = [(x[0], x[2]["label"]) for x in arg_edges]

        return len(set(heads)) == len(heads)

    def decode_constrained(self, G, feats, scorer, fmaps):

        for iteration in range(self.lr_maxiter):
            # if not iteration % 100:
            #     logger.info("Lagrangian Relaxation. Iteration %d" % (iteration))

            E_neg = []  # E_neg ($e \in E | \theta^{T} g(e) \leq 0$) - prioritize by scores

            heads, non_vars = self.group_nodes(G)

            for tail in non_vars:
                self.decode_1d(heads, tail, feats, G, scorer, fmaps)

            for node_pair in itertools.combinations(heads, 2):
                nid1, nid2 = node_pair

                # -----------------------------------------------
                # scoring 1 direction

                x1, y1 = feats[nid1][nid2]
                x_ind, x_data = x1.indices, x1.data
                scores1 = scorer(x_ind, x_data)

                # TODO adding constraints here
                for role_id in self.arg_ids:
                    k = self.mu2id[(nid1, role_id)]
                    scores1[role_id] -= self.mu[k]  # fixme: is it plus or minus?

                sorted_labids1 = np.argsort(scores1, axis=None)
                l12, l11 = sorted_labids1[-2:]
                s11, s12 = scores1[l11], scores1[l12]

                e11 = (nid1, nid2, {"label": fmaps["ri_i2l"][l11]})
                e12 = (nid1, nid2, {"label": fmaps["ri_i2l"][l12]})

                # -----------------------------------------------
                # scoring 2 direction

                x2, y2 = feats[nid2][nid1]
                x_ind, x_data = x2.indices, x2.data
                scores2 = scorer(x_ind, x_data)

                # TODO add constraints here
                for role_id in self.arg_ids:
                    k = self.mu2id[(nid2, role_id)]
                    scores2[role_id] -= self.mu[k]  # fixme: is it plus or minus?

                sorted_labids2 = np.argsort(scores2, axis=None)
                l22, l21 = sorted_labids2[-2:]
                s21, s22 = scores2[l21], scores2[l22]

                e21 = (nid2, nid1, {"label": fmaps["ri_i2l"][l21]})
                e22 = (nid2, nid1, {"label": fmaps["ri_i2l"][l22]})

                if l11 == 0:
                    E_neg.append((e12, s12))
                    if l21 == 0:
                        E_neg.append((e22, s22))
                    else:
                        G.add_edge(*e21)

                elif l21 != 0:
                    if s11 > s21:
                        G.add_edge(*e11)
                    else:
                        G.add_edge(*e21)
                else:
                    G.add_edge(*e11)

            if len(G.edges()) == 0 or len(E_neg) == 0:
                return

            self.MSCG(G, E_neg)

            # check if we satisfy the constraints
            if self.check_determinism(G):
                # logger.info("Converged after %i iterations" % (iteration))
                self.success += 1
                return

            for triple in G.edges(data=True):
                head, tail, role = triple[0], triple[1], fmaps["ri_l2i"][triple[2]["label"]]
                if role in self.arg_ids:
                    key = (head, role)
                    k = self.mu2id[key]
                    self.A[k] += 1  # this is essentially Az

                subg = self.b - self.A  # fixme plus or minus?
                self.mu = np.maximum(0.0, self.mu - self.lr_stepsize * subg)

        if not self.check_determinism(G):
            # logger.info("Lagrangian Relaxation failed!")
            self.failed += 1

    # ==================================================================

    def check_edge_determinism(self, all_edges, nid):

        nid_edges = filter(lambda e: e[0] == nid, all_edges)
        arg_edges = [x[2]["label"] for x in filter(lambda e: e[2]["label"] in self.arg_elabs, nid_edges)]
        uniq_arg_edges = list(set(arg_edges))

        # if there are more outgoing edges with semantic arg,
        # edge (and hence G) is not deterministic
        if len(arg_edges) != len(uniq_arg_edges):
            for triple in arg_edges:
                head, tail, role = triple[0], triple[1], triple[2]["label"]
                key = (head, role)
                idx = self.mu2id[key]
                self.A[idx] += 1  # this is essentially Az
            return False
        else:
            return True

    def group_nodes(self, G):

        # Split all nodes into 2 groups:
        # 1) w/ vars
        # 2) w/o vars

        heads, non_vars = [], []

        for nid in G.nodes():
            if G.node[nid]["var"] is None:
                non_vars.append(nid)
            else:
                heads.append(nid)
        return heads, non_vars

    def decode_1d(self, heads, tail, feats, G, scorer, fmaps):

        """
        Decode an edge between a pair of nodes. ONE DIRECTION
        This is the case of a relation between nodes which do not have
        any variables (NE, numbers) and node with variables

        """
        # assign one head for each of the non-var nodes

        cands = []
        for nid1 in heads:
            x, y = feats[nid1][tail]
            x_ind, x_data = x.indices, x.data
            scores = scorer(x_ind, x_data)
            sorted_labids = np.argsort(scores, axis=None)
            l12, l11 = sorted_labids[-2:]
            s11, s12 = scores[l11], scores[l12]
            if l11 == 0:
                cands.append((nid1, l12, s12))
            else:
                cands.append((nid1, l11, s11))

        best_cand = max(cands, key=operator.itemgetter(2))
        head, lab = best_cand[:2]
        G.add_edge(head, tail, {"label": fmaps["ri_i2l"][lab]})
