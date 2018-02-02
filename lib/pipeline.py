#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os,sys
import random
import logging
import json
import cPickle as pickle
import time, pprint
import numpy as np

import networkx as nx
import collections, itertools
import copy
import string

import util
from eval.prf import compute_prf, compute_prf2
from amr_printer import Printer
from decode import Decoder
from sklearn.metrics import confusion_matrix as cm
from lib.fxtract import Fxtractor
from lib.classifiers import Perceptron
from lib.rules import Ruleset


logger = logging.getLogger("main")


class Stage(object):
    def __init__(self, params, stage_name):

        self.params = params
        self.name = stage_name
        self.errors = collections.defaultdict()

        self.fx = None
        self.clf = None
        self.ruleset = None
        self.amr_printer = None
        self.decoder = None

    def train(self):
        pass

    def predict_train(self, data, print_stats=False):
        pass

    def predict_test(self, data):
        pass

    def evaluate(self, data, save=False):
        pass

    def save_results(self, *args):
        pass

    def load_model(self, model_fname):
        pass


class CI_Stage(Stage):

    def train_setup(self):

        from featlists.ci import feats as flist

        assert "train_pkl" in self.params and "dev_pkl" in self.params
        logger.info(" * TRAINING: %s stage * " % self.name)

        vocab = util.load_pkl(self.params["vocab_fn"])
        train_data = util.load_pkl(self.params["train_pkl"])
        val_data = util.load_pkl(self.params["dev_pkl"])
        test_data = util.load_pkl(self.params["test_pkl"]) if "test_pkl" in self.params else None

        self.fx = Fxtractor(fx_maps=collections.defaultdict(), labels=vocab["ctypes"], flist=flist)
        self.fx.fx_dataset_ci(train_data, train=True)
        self.fx.fx_dataset_ci(val_data, train=False)
        self.fx.fx_maps["vidx"] = copy.deepcopy(vocab["vidx"])

        n_features = self.fx.fx_maps["ci_fdim"]
        n_classes = len(self.fx.fx_maps["ci_l2i"])
        self.clf = Perceptron.SparsePerceptron()
        self.clf.init_w(n_features, n_classes)

        self.ruleset = Ruleset()
        self.ruleset.load_db()

        logger.debug("Data: %d (train), %d (dev). Num feats %d" % (len(train_data), len(val_data), n_features))
        logger.debug("Used features: \n%s", "\n".join(self.fx.flist))

        return vocab, train_data, val_data, test_data

    def train(self):

        vocab, train_data, val_data, test_data = self.train_setup()

        for i in range(self.params["nepochs"]):
            
            random.seed(i)
            random.shuffle(train_data)
            logger.info("Iteration %d/%d" % (i + 1, self.params["nepochs"]))
            
            for idx, instance in enumerate(train_data):

                # if not idx % 1000:
                #     logger.info("Processing instance %d" % idx)

                self.train_one(instance)

            # evaluate on the dev set
            self.evaluate(val_data, section="dev", save=False)

        # average the parameters and evaluate on the dev set
        self.clf.avg_weights()
        self.evaluate(val_data, section="dev", save=False)

        # evaluate on the test set, if there is one
        if test_data is not None:
            self.fx.fx_dataset_ci(test_data, train=False)
            self.evaluate(test_data, section="test", save=True)

    def train_one(self, instance):

        tokens = instance["tokens"]

        # this works better than the sentence order like range(0, len(tokens)-1)
        token_order = nx.topological_sort(instance["depG"])

        for tokid in token_order:
            tok = tokens[tokid]
            word = tok["word"].lower()

            if tok["ctype"] == "misc":
                continue

            elif "http" in word or "www" in word:
                continue

            elif word[0] in string.punctuation + "''":
                continue

            x, y = tok["ci_data"]
            x_ind, x_data = x.indices, x.data
            scores = self.clf.compute_scores(x_ind, x_data)
            guess = np.argmax(scores)
            if guess != y:
                self.clf.update_weights(y, guess, x_ind, x_data)

    def predict_train(self, data, print_stats=False):

        i2l = self.fx.fx_maps["ci_i2l"]
        l2i = self.fx.fx_maps["ci_l2i"]
        vidx = self.fx.fx_maps["vidx"]

        golds, preds = [], []
        y_pred, y_true = [], []
        total = correct = 0.0

        for instance_idx, instance in enumerate(data):

            tokens = instance["tokens"]
            token_order = nx.topological_sort(instance["depG"])
            gG = instance["amrG"]
            pG = nx.DiGraph(maxid=-1, varnum=-1, comments=instance["comments"])

            for tokid in token_order:
                tok = tokens[tokid]
                word = tok["word"]
                pos = tok["pos"]
                base_form = self.ruleset.get_tokbase(tok)

                if word[0].isdigit():
                    self.ruleset.fill_digit_ctype(pG, tokens, tokid, tok, word, pos)

                elif word[0] in string.punctuation + "''":
                    continue

                elif "http" in word or "www" in word:
                    self.ruleset.fill_url_concept(pG, tokid, word)

                elif base_form in self.ruleset.main_dict:
                    concepts = self.ruleset.main_dict[base_form]
                    self.ruleset.fill_misc_concepts(pG, tokid, tok, concepts)

                else:
                    x, y = tok["ci_data"]
                    x_ind, x_data = x.indices, x.data
                    scores = self.clf.compute_scores(x_ind, x_data)

                    legal_ids = [l2i[l] for l in self.ruleset.get_legal_ctype(tok)]
                    best_id = max(legal_ids, key=lambda l: scores[l])
                    best_lab = i2l[best_id]
                    self.ruleset.fill_concepts_from_ctypes(tokens, tokid, tok, best_lab, pG, vidx)

                    total += 1
                    if best_id == y:
                        correct += 1

                    y_pred.append(best_lab)
                    y_true.append(i2l[y])

            golds.append([na["true_concept"] for n, na in gG.nodes(data=True)])
            preds.append([na["true_concept"] for n, na in pG.nodes(data=True)])

        if print_stats:
            # performance stats
            util.plot_cm(cm(y_true,y_pred), sorted(i2l.values()))

            acc = 100. * correct / total
            logger.debug("Action prediction accuracy: %0.4f" % acc)

        return golds, preds

    def predict_test(self, data):

        i2l = self.fx.fx_maps["ci_i2l"]
        l2i = self.fx.fx_maps["ci_l2i"]
        vidx = self.fx.fx_maps["vidx"]

        self.fx.fx_dataset_ci(data, train=False)
        logger.debug("Predicting on test")

        for idx, instance in enumerate(data):

            tokens = instance["tokens"]
            token_order = nx.topological_sort(instance["depG"])

            G = nx.DiGraph(maxid=-1, varnum=-1,comments=instance["comments"])
            back_off_candidates = []

            for tokid in token_order:

                tok = tokens[tokid]
                word = tok["word"].lower()
                pos = tok["pos"]
                base_form = self.ruleset.get_tokbase(tok)

                if word[0].isdigit():
                    self.ruleset.fill_digit_ctype(G, tokens, tokid, tok, word, pos)

                elif "http" in word or "www" in word:
                    self.ruleset.fill_url_concept(G, tokid, word)

                elif word[0] in string.punctuation+"''":
                    continue

                elif base_form in self.ruleset.main_dict:
                    concepts = self.ruleset.main_dict[base_form]
                    self.ruleset.fill_misc_concepts(G, tokid, tok, concepts)

                else:
                    x, _ = tok["ci_data"]
                    x_ind, x_data = x.indices, x.data
                    scores = self.clf.compute_scores(x_ind, x_data)
                    legal_ids = [l2i[l] for l in self.ruleset.get_legal_ctype(tok)]
                    best_id = max(legal_ids, key=lambda l: scores[l])
                    best_lab = i2l[best_id]
                    self.ruleset.fill_concepts_from_ctypes(tokens, tokid, tok, best_lab, G, vidx)
                    back_off_candidates.append((tok["lemma"], tokid))

            if len(G.nodes()) == 0:
                ad_hoc_concept = max(back_off_candidates, key=lambda a: len(a[0]))
                concept, tokid = ad_hoc_concept
                self.ruleset.fill_one_node(G,tokid,concept,var=True)

            instance["amrG"] = G

    def save_results(self, f_score, section):

        default_fn = os.path.join(self.params["model_dir"], "%s.%0.3f" % (section, f_score))
        base_fn = util.get_fname(default_fn)

        param_fn = "%s.params.json" % base_fn
        json.dump(self.params, open(param_fn, "w"))

        model_fn = "%s.CI_model.pkl" % base_fn
        with open(model_fn, "wb") as file_out:
            logger.info("Saving the model into -----> %s " % (model_fn))
            pickle.dump((self.clf.w, self.fx.fx_maps, self.fx.flist), file_out)

        # errors_fn = "%s.errors.json" % base_fn
        # json.dump(self.errors, open(errors_fn, "w"))

    def evaluate(self, data, section="dev", save=False):
        golds, preds = self.predict_train(data, print_stats=save)

        P, R, F1 = compute_prf2(golds, preds, stage=self.name, save=save, savedir=self.params["stat_dir"])
        logger.info("CI result on %s (2): %0.4f (P), %0.4f (R), %0.4f (F1)" % (section, P, R, F1))

        if save:
            self.save_results(F1, section)

    def load_model(self, model_fname):
        logger.info("Loading CI model")

        weights, feature_maps, feature_list = pickle.load(open(model_fname, "rb"))
        self.clf = Perceptron.SparsePerceptron()
        self.clf.w = weights

        self.fx = Fxtractor(fx_maps=feature_maps, flist=feature_list)
        self.ruleset = Ruleset()
        self.ruleset.load_db()

class RI_Stage(Stage):

    def train_setup(self):

        from featlists.ri import feats as flist

        assert "train_pkl" in self.params and "dev_pkl" in self.params
        logger.info(" * TRAINING: %s stage * " % self.name)

        vocab = util.load_pkl(self.params["vocab_fn"])
        train_data = util.load_pkl(self.params["train_pkl"])
        val_data = util.load_pkl(self.params["dev_pkl"])
        test_data = util.load_pkl(self.params["test_pkl"]) if "test_pkl" in self.params else None

        labels = ["<null>", "<unk>"] + vocab["edges"].keys()
        self.fx = Fxtractor(fx_maps=collections.defaultdict(), labels=labels, flist=flist)
        self.fx.fx_dataset_ri(train_data, train=True)
        self.fx.fx_dataset_ri(val_data, train=False)

        n_features = self.fx.fx_maps["ri_fdim"]
        n_classes = len(self.fx.fx_maps["ri_l2i"])
        self.clf = Perceptron.SparsePerceptron()
        self.clf.init_w(n_features, n_classes)

        self.amr_printer = Printer()

        logger.debug("Data: %d (train), %d (dev). Num feats %d" % (len(train_data), len(val_data), n_features))
        logger.debug("Used features: \n%s", "\n".join(self.fx.flist))

        return vocab, train_data, val_data, test_data

    def train(self):

        vocab, train_data, val_data, test_data = self.train_setup()

        for i in range(self.params["nepochs"]):

            random.seed(i)
            random.shuffle(train_data)

            logger.info("Iteration %d/%d" % (i + 1, self.params["nepochs"]))

            for idx, instance in enumerate(train_data):

                # if not idx % 1000:
                #     logger.info("Processing instance %d" % idx)

                self.train_one(instance)

            # evaluation on dev set
            self.evaluate(val_data, section="dev",save=False)

        # average the parameters and evaluate on the dev set
        self.clf.avg_weights()
        self.evaluate(val_data, section="dev", save=True)

        # evaluate on the test set, if there is one
        if test_data is not None:
            self.fx.fx_dataset_ri(test_data, train=False)
            self.evaluate(test_data, section="test", save=True)

    def get_dephead_token(self, nid, aG, toposort):
        start, end = aG.node[nid]["span"]
        tokid_range = end - start
        if tokid_range == 1:
            head_tokid = start
            head_idx = toposort.index(head_tokid)

        else:
            tokids = range(start, end)
            head_idx = max([toposort.index(i) for i in tokids])
            head_tokid = toposort[head_idx]

        return (head_idx, head_tokid)

    def train_one(self, instance):

        amrG = instance["amrG"]
        tokens = instance["tokens"]
        gold_nodes = filter(lambda x: amrG.node[x]["span"][0] is not None, amrG.nodes())
        ri_data = instance["ri_data"]

        # A = [" ".join(T["concepts"][1:]) for T in tokens if len(T["node_id"]) > 1 and "name" not in T["concepts"][1:]]
        # if len(A) > 0:
        #     pprint.pprint(A)

        for node_pair in itertools.combinations(gold_nodes, 2):

            # score each candidate fx and find the max one
            nid1, nid2 = node_pair

            # first direction
            x1, y1 = ri_data[nid1][nid2]
            x_ind, x_data = x1.indices, x1.data
            scores1 = self.clf.compute_scores(x_ind, x_data)
            best_clas1 = np.argmax(scores1)
            if best_clas1 != y1:
                self.clf.update_weights(y1, best_clas1, x_ind, x_data)

            # second direction
            x2, y2 = ri_data[nid2][nid1]
            x_ind, x_data = x2.indices, x2.data
            scores2 = self.clf.compute_scores(x_ind, x_data)
            best_clas2 = np.argmax(scores2)
            if best_clas2 != y2:
                self.clf.update_weights(y2, best_clas2, x_ind, x_data)

    def predict_train(self, data, print_stats=False):

        golds, preds, graphs = [], [], []
        fmaps = self.fx.fx_maps
        scorer = self.clf.compute_scores

        # initialize the decoder
        arg_labels = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5",
                      # "ARG0-of", "ARG1-of", "ARG2-of", "ARG3-of", "ARG4-of", "ARG5-of",
                      ]

        arg_ids = [self.fx.fx_maps["ri_l2i"][l] for l in arg_labels]
        self.decoder = Decoder(arg_elabs=arg_labels, arg_ids=arg_ids)

        for idx, instance in enumerate(data):

            goldG = instance["amrG"]
            feats = instance["ri_data"]
            aligned_nodes = filter(lambda x: goldG.node[x[0]]["span"][0] is not None, goldG.nodes(data=True))
            G = nx.DiGraph(root=None, comments=instance["comments"])

            if len(aligned_nodes) == 0:
                snt = " ".join([T["word"] for T in instance["tokens"]])
                print "No aligned nodes in AMR for snt (%s): %s" % (instance["comments"]["id"], snt)

                # tokid = 0
                # random_concept = instance["tokens"][tokid]["lemma"]
                # self.ruleset.fill_one_node(G, tokid, random_concept, var=True)
                # graphs.append(G)
                # fixme: in JAMR, they exclude these graphs at all?
                goldG.graph["comments"] = instance["comments"]
                graphs.append(goldG)
                continue

            G.add_nodes_from(aligned_nodes)
            # In Flanigan et al., these are edges from the CI stage to be preserved
            # we disregard them, as there are hardly any
            # E_0 = G.edges(data=False)

            self.decoder.decode_one(feats, G, scorer, fmaps)

            preds.append([(e[0], e[1], e[2]["label"]) for e in G.edges(data=True)])
            golds.append([(e[0], e[1], e[2]["label"]) for e in goldG.edges(data=True)])
            graphs.append(G)

        # print "ND graphs decoding (+/-): %d/%d" %(self.decoder.success, self.decoder.failed)
        self.decoder.success = self.decoder.failed = 0
        return golds, preds, graphs

    def predict_test(self, data,evaluate=False):

        graphs = []
        fmaps = self.fx.fx_maps
        scorer = self.clf.compute_scores

        self.fx.fx_dataset_ri(data, train=False)

        # initialize the decoder
        arg_labels = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5",
                      # "ARG0-of", "ARG1-of", "ARG2-of", "ARG3-of", "ARG4-of", "ARG5-of",

                      ]

        arg_ids = [self.fx.fx_maps["ri_l2i"][l] for l in arg_labels]
        self.decoder = Decoder(arg_elabs=arg_labels, arg_ids=arg_ids)

        for idx, instance in enumerate(data):
            feats = instance["ri_data"]

            # we have a partially built G from the CI stage
            # no nodes are unaligned
            G = instance["amrG"]
            if len(G.nodes()) == 0:
                snt = " ".join([T["word"] for T in instance["tokens"]])
                print "No aligned nodes! Sentence (%s): %s" % (instance["comments"]["id"], snt)
                print "This should not happen!"
                sys.exit()

                tokid = 0
                random_concept = instance["tokens"][tokid]["lemma"]
                self.ruleset.fill_one_node(G, tokid, random_concept, var=True)
                graphs.append(G)
                continue

            self.decoder.decode_one(feats, G, scorer, fmaps)
            graphs.append(G)

        return graphs

    def save_results(self, f_score, section, graphs, save_graphs=False):

        default_fn = os.path.join(self.params["model_dir"], "%s.%0.3f" % (section, f_score))
        base_fn = util.get_fname(default_fn)

        amr_fn = "%s.amr" % base_fn
        edge_fn = "%s.edges" % base_fn
        self.amr_printer.save_penman(graphs, amr_fn, edge_fn)

        param_fn = "%s.params.json" % base_fn
        with open(param_fn, "w") as param_out:
            json.dump(self.params, param_out)

        if save_graphs:
            graph_fn = "%s.graphs.pkl" % base_fn
            self.amr_printer.pkl_graphs(graphs, graph_fn)

        model_fn = "%s.ri_weights.pkl" % base_fn
        with open(model_fn, "wb") as file_out:
            logger.info("Saving the model into -----> %s " % (model_fn))
            pickle.dump((self.clf.w, self.fx.fx_maps, self.fx.flist), file_out)

        errors_fn = "%s.errors.json" % base_fn
        with open(errors_fn, "w") as errors_out:
            json.dump(self.errors, errors_out)

    def load_model(self, model_fname):
        logger.info("Loading RI model")

        weights, feature_maps, feature_list = pickle.load(open(model_fname, "rb"))
        self.clf = Perceptron.SparsePerceptron()
        self.clf.w = weights
        self.fx = Fxtractor(fx_maps=feature_maps, flist=feature_list)
        self.amr_printer = Printer()

    def evaluate(self, data, section="dev", save=False):
        golds, preds, graphs = self.predict_train(data)
        P, R, F1 = compute_prf(golds, preds, stage=self.name, save=save, savedir=self.params["stat_dir"])
        logger.info("RI result on %s (1): %0.4f (P), %0.4f (R), %0.4f (F1)" % (section, P, R, F1))

        if save:
            self.save_results(F1, section, graphs, save_graphs=False)

def parse_all_stages(params):

    # need 2 model files and one data file (.pkl)
    assert len(params["model_fn"]) == 2

    ci_model_fn, ri_model_fn = params["model_fn"]
    data_fn = params["test_pkl"]
    data = pickle.load(open(data_fn, "rb"))

    # CI prediction
    ci_parser = CI_Stage(params, stage_name="CI")
    ci_parser.load_model(model_fname=ci_model_fn)
    ci_parser.predict_test(data)

    # RI prediction
    ri_parser = RI_Stage(params=params, stage_name="RI")
    ri_parser.load_model(model_fname=ri_model_fn)
    amr_graphs = ri_parser.predict_test(data)

    # save the results
    base_fn = os.path.abspath(os.path.join(params["model_dir"], "%s.parse" % os.path.split(data_fn)[1]))
    amr_fn = "%s.amr" % base_fn
    edge_fn = "%s.edges" % base_fn
    ri_parser.amr_printer.save_penman(amr_graphs, amr_fn, edge_fn)