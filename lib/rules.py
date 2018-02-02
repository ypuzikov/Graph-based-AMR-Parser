#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import re
import json
import random
import collections
import string
from pattern.en import singularize as sng
from pyjarowinkler import distance
import dateutil.parser
import pprint
import copy
import cPickle as pickle
import time
import logging

logger = logging.getLogger("main")

def rec_dd():
    return collections.defaultdict()

ordinal_to_number_map = {
    '0th': '0',
    '1st': '1',
    '2nd': '2',
    '3rd': '3',
    '4th': '4',
    '5th': '5',
    '6th': '6',
    '7th': '7',
    '8th': '8',
    '9th': '9',
    '11th': '11',
    '12th': '12',
    '13th': '13',
}

ctypes = ["lemma",
          "empty",
          "sng",
          "ly",
          "comp_adj",
          "sup_adj",
          "adj2n",
          "adv2n",
          "verb",
          "org", "person", "country", "unk_ne",

          "person-verb",
          "thing-verb",


          # "ordinal",, "digit", "listitem"
          # "org-role",
          # "possibility",

          # "misc",

          ]

# ============================================  PATTERNS

verb_pat = re.compile(r"\w+-.*?\d+")
date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
decade_pat1 = re.compile(r"^\d{4}s$")
decade_pat2 = re.compile(r"^\d{2}s$")
decade_pat3 = re.compile(r"^\'\d0s$")
date_interval_pat = re.compile(r"^\d{4}-\d{4}$")

digit_pat = re.compile(r"\d+[\s\.,]?(\d+)?")
var_pat = re.compile(r"((\()?([a-z]+[0-9]?[0-9]?)(\s/\s)(.+?)[\s\n\)])")  # groups 3 (var) and 5 (concepts)
ordinal_num_pat = re.compile(r'((?:^|\W)(?:\d*))(' + '|'.join(ordinal_to_number_map.keys()) + ')(?=\W|$)', re.I)

# ============================================  DATABASES

resource_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))
demonym_fn = os.path.join(resource_dir, "db/nations.json")
cvb_fn = os.path.join(resource_dir, "db/CVB.json")

# ============================================ WORDLISTS

verb_index_fn = os.path.join(resource_dir,"wlists", "verblist.index.json")
pverbL_fn = os.path.join(resource_dir,"wlists", "prop_verblist")
conceptL_fn = os.path.join(resource_dir,"wlists", "concept_list")

orgL_fn = os.path.join(resource_dir,"wlists", "have-org-role-91-roles-v1.01.txt")
relL_fn = os.path.join(resource_dir,"wlists", "have-rel-role-91-roles-v1.01.txt")
adjL_fn = os.path.join(resource_dir,"wlists", "comparatives_superlatives.txt")
adj41L_fn = os.path.join(resource_dir,"wlists", "41.txt")
ly2nL_fn = os.path.join(resource_dir,"wlists", "ly2n.txt")
miscL_fn = os.path.join(resource_dir,"wlists", "misc")
iverbs_dir = os.path.join(resource_dir,"wlists", "iverbs")
hyphL_fn = os.path.join(resource_dir,"wlists", "hyphenated_concepts.txt")
quantL_fn = os.path.join(resource_dir,"wlists", "quant.txt")

emptyL_fn = os.path.join(resource_dir,"wlists", "empty")

# ============================================

class Ruleset(object):
    def __init__(self):

        self.patterns = {"verb": verb_pat,
                         "digit": digit_pat,
                         "var": var_pat,
                         "ordnum": ordinal_num_pat,
                         "date": date_pat,
                         "dec1": decade_pat1,
                         "dec2": decade_pat2,
                         "dec3": decade_pat3,
                         "date_interv": date_interval_pat,

                         }

        self.errors = 0
        self.ctypes = ctypes
        self.concept_list = []
        self.hyph_dict = []
        self.hyph_dict_prefix = []
        self.empties = []


        self.misc_list = []
        self.misc_dict = {}
        self.misc_num = 0

    # ============================================ LOAD DB

    def load_db(self, main_dict_fn = None):

        self.load_cvb()
        self.load_irregular_verbdict()
        # self.load_concept_list()

        if main_dict_fn is None:
            logger.info("No concept table specified. Creating one using wordlists.")
            self.main_dict = {}
            self.gen_main_dict()

        else:
            self.main_dict = json.load(open(main_dict_fn))

        logger.info("Loaded concept table (%d items)" %(len(self.main_dict)))

    def load_concept_list(self, fn = conceptL_fn):

        """
        Load a list of concepts, extracted from training data

        """

        with open(fn) as infile:
            for line in infile:
                element = line.strip()
                if not element[0].isdigit() and element[0] != '"':
                    self.concept_list.append(element.lower())

    def load_demonyms(self, fname=demonym_fn):
        data = json.load(open(fname))
        for k ,v in data.items():
            key = k.lower()
            triples = {}
            triples.setdefault("country",[]).append(("name", "name"))
            for val in v.split():
                triples.setdefault("name", []).append((val, "op"))
            self.misc_dict[key] = triples
            # print key, self.misc_dict[key]
            # time.sleep(1)


    def load_cvb(self, fname=cvb_fn):
        self.cvb = json.load(open(fname))

    def load_org(self, org_fn):
        # fixme: don't forget about the triple: ("person", "have-org-role-91", "ARG0-of")
        # have-org-role-91@ARG2@word-01
        with open(org_fn) as ofl:
            for line in ofl:
                if line.startswith("#") or not line.strip():
                    continue

                base, triple = line.strip().split()
                hc, role, tc = triple.split("@")
                self.main_dict[base] = (hc, tc, role)

    def load_rel(self, rel_fn=relL_fn):
        with open(rel_fn) as rfl:
            for line in rfl:
                if line.startswith("#"):
                    continue
                word = line.strip()
                if word in self.main_dict:
                    continue
                self.main_dict[word] = ("have-rel-role-91", word, "ARG2")

    def load_comp_adj(self, adj_degree_fn):

        # word@degree@more or word@degree@most
        with open(adj_degree_fn) as dfl:
            for line in dfl:
                if line.startswith("#"):
                    continue

                base, comp, sup = line.strip().split("\t")
                assert comp not in self.main_dict
                assert sup not in self.main_dict

                if base in self.main_dict:
                    # print "Base for comparative/superlative is already in the dict: %s" %base
                    base = "%s-41" %base
                    self.main_dict[comp] = (base, "more", "degree")
                    self.main_dict[sup] = (base, "most", "degree")
                    continue

                self.main_dict[comp] = (base, "more", "degree")
                self.main_dict[sup] = (base, "most", "degree")

    def load_41(self, fn):
        with open(fn) as infile:
            for line in infile:
                item = line.strip()
                self.main_dict[item] = "%s-41" % item
                # self.main_dict[item] = "%s-00" % item
                self.misc_list.append(item)

    def load_ly(self, ly_fn):

        # adv(-ly) -> ?@mod@some_word
        # a verb/noun/adj
        with open(ly_fn) as lfl:
            for line in lfl:
                if line.startswith("#"):
                    continue
                adv, base = line.strip().split()
                assert adv not in self.main_dict, "%s" % adv
                self.main_dict[adv] = base
                # TODO: self.main_dict[adv] = (base, "mod")

    def load_irregular_verbdict(self, ivdir=iverbs_dir):

        self.irr_verbs = {}

        for fn in os.listdir(ivdir):
            fname = os.path.join(ivdir, fn)
            with open(fname) as infile:
                for line in infile:
                    elems = line.strip().split("-")
                    base, past, ppast = elems[:3]
                    for item in [past, ppast]:
                        self.irr_verbs[item] = base

    def load_hyph(self, fname = hyphL_fn):
        with open(fname) as infile:
            for line in infile:
                item = line.strip()
                prefix = item.split("-")[0]
                self.hyph_dict.append(line.strip())
                self.hyph_dict_prefix.append(prefix)

    def load_quant(self, fname=quantL_fn):
        with open(fname) as infile:
            for line in infile:
                item = line.strip()
                word, quant_concept = item.split()
                self.main_dict[word] = (quant_concept, word, "unit")

    def load_empty(self, fname = emptyL_fn):
        with open(fname) as infile:
            for line in infile:
                if not line.strip() or line.startswith("#"):
                    continue
                item = line.strip()
                self.empties.append(item)

    def load_misc(self, misc_fn = miscL_fn):

        with open(misc_fn) as mfl:
            for line in mfl:
                if line.startswith("#") or not line.strip():
                    continue

                elems = line.strip().split()
                key = elems[0]
                assert key not in self.main_dict, key
                # -------------------------------------------------
                if len(elems) == 2:
                    triple = elems[1].split("@")

                    if len(triple) == 1:  # no  -
                        self.main_dict[key] = elems[1]

                    elif len(triple) == 3:  # e.g., insufficient    suffice@polarity@-
                        head_concept, relation, tail_concept = triple
                        self.main_dict[key] = (head_concept, tail_concept, relation)

                    else:
                        print "UNKNOWN FORMAT in misc list: %s" % elems
                        sys.exit()
                # -------------------------------------------------
                elif len(elems) == 3:  # e.g., unbelievable possible@domain@believe-01 possible@polarity@-

                    # concepts = []
                    #
                    # try:
                    #     head_concept1, relation1, tail_concept1 = elems[1].split("@")
                    # except ValueError:
                    #     print "@ elements count error: ", elems[1]
                    #     sys.exit()
                    #
                    # concepts.append(head_concept1)
                    # concepts.append((tail_concept1, relation1))
                    # for elem in elems[2:]:
                    #     head_concept, relation2, tail_concept2 = elem.split("@")
                    #     assert head_concept == head_concept1, "%s %s" % (head_concept, head_concept1)
                    #     concepts.append((tail_concept2, relation2))
                    #
                    # self.neg[key] = concepts
                    # self.main_dict[key] = concepts

                    triples = {}
                    for e in elems[1:]:
                        head_concept, relation, tail_concept = e.split("@")
                        triples.setdefault(head_concept, []).append((tail_concept, relation))

                    self.main_dict[key] = triples
                # -------------------------------------------------
                else:
                    print "UNKNOWN FORMAT in misc list: %s" % elems
                    sys.exit()

    # ============================================ GENERATE DB

    @staticmethod
    def gen_propbank_vlist(fn=pverbL_fn):

        """
        Build a dictionary of the form {verb: [sense1, sense2, sense3]}

        """

        L = {}

        with open(fn) as infile:
            for line in infile:
                items = line.strip().split("-")

                if len(items) > 2:
                    continue

                verb = items[0]
                sense = items[-1]
                L.setdefault(verb, []).append(sense)

        logger.info("Generated Propbank verblist: %d verbs" % (len(L)))
        random_key = random.choice(L.keys())
        logger.debug("Random sample: %s %s " % (random_key,L[random_key]))

        return L

    def gen_main_dict(self, save=False):

        self.load_org(orgL_fn)
        self.load_41(adj41L_fn)
        self.load_comp_adj(adjL_fn)
        self.load_ly(ly2nL_fn)
        self.load_hyph()
        self.load_quant()
        self.load_misc()

        # self.load_demonyms()
        # self.load_empty()
        # self.load_rel()

        # these lists can be used but not needed #
        # self.load_misc(miscL_fn)
        # self.load_morph_verb(morph_fn)
        # self.load_verbalization(verb_fn)
        # usnig my own verbalization list (based on Ulf)
        # self.load_rel(rel_fn)
        # self.load_neg(neg_fn)
        # load_irregular_verbs(ivd, main_dict)

    def save_main_dict(self, main_dictionary_fname):
        logger.info("Saving the main dict into --> %s" %(main_dictionary_fname))
        with open(main_dictionary_fname,"w") as out:
            json.dump(self.main_dict,out)

    # ============================================

    def fill_netype(self, amrG, token):

        # below is an important procedure
        # to find out the gold type of NE

        start_N = token["node_id"][-1]

        while True:
            try:
                pid = amrG.predecessors(start_N)[0]  # check a node upper in the graph
                pid_concept = amrG.node[pid]["concept"]
            except IndexError:
                # in rare cases "name" is the NE to use ("alias")
                # we use it only if start_N is a root of the graph
                ne_type = amrG.node[start_N]["concept"]
                token["netype"] = ne_type
                break

            # add the concept if it is not "name", "verb-01" or another NE
            if pid_concept != "name":
                if not pid_concept.startswith('"'):

                    if re.match(r"\w+-\d+", pid_concept) is None:
                        ne_type = pid_concept
                        token["netype"] = ne_type
                        break

                    else:
                        # sometimes the NE we are looking for is a sibling of "name"
                        # so when we hit the verb in the tree,
                        # we go 1 level down and take its first child as NE
                        ne_type = amrG.node[amrG.successors(pid)[0]]["concept"]
                        token["netype"] = ne_type
                        break

            start_N = pid

        return

    def get_tokbase(self, tok):

        pos = tok["pos"]
        lem = tok["lemma"].lower()
        word = tok["word"].lower()

        nform = lem

        # noun-plr
        if pos == "NNS":
            nform = sng(word, pos="NOUN")

        # adjectives
        elif pos.startswith("J") and word[0].isalpha():
            nform = word

        return nform

    def get_verbbase(self, base, pos):

        # if the past form is irregular verb - look up in the dict
        if base in self.irr_verbs:
            base = self.irr_verbs[base]

        # if gerund
        elif base[-3:] == "ing" and len(base) >= 5:
            if base != "bring":
                base = base[:-3]

        return base

    def get_concept_from_neg_form(self, token):

        def check_cvb(full_form, pos, reduced_form, pref_len):

            if pos == "JJ" and reduced_form in self.cvb["AJ"]:  # adj -> ?
                return check_cvb_AJ(full_form, reduced_form)

            elif pos in ["NN", "NNS", "VBG"]:
                sng_form = sng(full_form)
                reduced_form = sng_form[pref_len:]
                if reduced_form in self.cvb["N"]:
                    return check_cvb_N(lem, reduced_form)  # noun -> ?
                else:
                    return [False, sng_form]

            # try to find if there is a verb without the prefix
            elif reduced_form in self.cvb["V"] and "V" in self.cvb["V"][reduced_form]:  # verb -> verb
                return [True, reduced_form]

            elif pos == "RB" and reduced_form in self.cvb["AV"]:
                if "V" in self.cvb["AV"][reduced_form]:
                    return [True, self.cvb["AV"][reduced_form]["V"][0]]  # adv -> verb

                elif "AJ" in self.cvb["AV"][reduced_form]:
                    return [True, self.cvb["AV"][reduced_form]["AJ"][0]]  # adv -> adj

                else:
                    return [False, full_form]

            else:
                return [False, full_form]

        def check_cvb_AJ(full_form, reduced_form):

            if reduced_form not in self.cvb["AJ"]:
                return [False, full_form]

            elif reduced_form[-2:] == "al":  # most likely derived adj -> noun
                return [True, reduced_form]
            elif reduced_form[-3:] == "ant":  # most likely adj -> verb / adj -> adj
                return [True, reduced_form]

            # elif len(reduced_form) > 5 and reduced_form[-4:] == "able":
            #
            #     if "V" in self.cvb["AJ"][reduced_form]:                 # adj(able) -> verb
            #         cands = self.cvb["AJ"][reduced_form]["V"]
            #         # best_cand = cands[0]
            #         best_cand = max(cands, key=lambda x:len(x))
            #         return [True, best_cand]  # verbal form
            #
            #     if "AJ" in self.cvb["AJ"][reduced_form]:            # adj(able) -> verb
            #         cands = self.cvb["AJ"][reduced_form]["AJ"]
            #         best_cand = cands[0]
            #         # best_cand = max(cands, key=lambda x: len(x))
            #         return [True, best_cand]
            #
            #     else:
            #         return ["UNK", full_form]

            elif "AJ" not in self.cvb["AJ"][reduced_form] and "V" not in self.cvb["AJ"][reduced_form]:
                return [True, reduced_form]

            elif "AJ" in self.cvb["AJ"][reduced_form] and "V" not in self.cvb["AJ"][reduced_form]:  # adj -> adj
                cands = self.cvb["AJ"][reduced_form]["AJ"]
                best_cand = cands[0]
                # best_cand = max(cands, key=lambda x: len(x))
                return [True, best_cand]

            elif "V" in self.cvb["AJ"][reduced_form]:  # adj -> verb
                cands = self.cvb["AJ"][reduced_form]["V"]
                # best_cand = cands[0]
                best_cand = max(cands, key=lambda x: len(x))
                return [True, best_cand]

            elif "N" in self.cvb["AJ"][reduced_form]:  # adj -> noun
                cands = self.cvb["AJ"][reduced_form]["N"]
                # best_cand = cands[0]
                best_cand = max(cands, key=lambda x: len(x))
                return [True, best_cand]

            else:
                return [True, reduced_form]  # adj -> adj -un

        def check_cvb_N(full_form, reduced_form):

            if "V" in self.cvb["N"][reduced_form]:
                cands = self.cvb["N"][reduced_form]["V"]
                best_cand = cands[0]
                # best_cand = max(cands, key=lambda x: len(x))
                return [True, best_cand]

            elif "AJ" in self.cvb["N"][reduced_form]:
                cands = self.cvb["N"][reduced_form]["AJ"]
                # best_cand = cands[0]
                best_cand = max(cands, key=lambda x: len(x))
                return [True, best_cand]

            elif "N" in self.cvb["N"][reduced_form]:
                # need to be careful!
                cands = self.cvb["N"][reduced_form]["N"]
                # best_cand = cands[0]
                best_cand = max(cands, key=lambda x: len(x))

                if "AJ" in self.cvb["N"][best_cand] and "AJ" in self.cvb["N"][full_form]:
                    adj1 = self.cvb["N"][best_cand]["AJ"][0]
                    adj2 = self.cvb["N"][full_form]["AJ"][0]
                    if adj1[-3:] == adj2[-3:]:
                        return [True, self.cvb["N"][reduced_form]["N"][0]]
                    else:
                        return [False, full_form]
                else:
                    return [False, full_form]

            else:
                return [False, full_form]

        # negative prefixes for English:
        # a–, dis–, il–, im–, in-, ir–, non–, un–

        lem = token["lemma"]
        pos = token["pos"]

        pref3 = lem[:3]
        pref2 = lem[:2]
        pref1 = lem[0]

        # # TODO fix -ly
        # # an AV formed by adding -ly to an adjective
        # form1 = token["lemma"][:-2]
        #
        # # if the AJ ends with "-y", we can create an AV
        # # by replacing the "-y" with "i" and adding "ly"
        # form2 = "%sy" % token["lemma"][:-3]
        #
        # # if the AJ ends with "-able", "-ible", or "-le", we can create an AV
        # # by replacing  the "-e" with "-y".
        # form3 = "%se" % (token["lemma"][:-1])
        #
        # # if the AJ ends with "-ic" (except "public"),
        # # we can crete an AV by adding "-ally"
        # form4 = token["lemma"][:-4]
        #
        # if token["lemma"] in self.ly2n:
        #     return self.ly2n[token["lemma"]]

        # 1. prefixes dis- and non-
        if pref3 == "non":
            reduced_form = lem[3:]
            return check_cvb(lem, pos, reduced_form, pref_len=3)

        elif pref3 == "dis":
            # dis is treated differently for some reason - not considered as a negative polarity
            if pos in ["NN", "NNS", "VBG"]:
                reduced_form = sng(lem)
                if reduced_form in self.cvb["N"] and "V" in self.cvb["N"][reduced_form]:
                    cands = self.cvb["N"][reduced_form]["V"]
                    # best_cand = cands[0]
                    best_cand = max(cands, key=lambda x: len(x))
                    return [False, best_cand]

                else:
                    return [False, lem]

            elif pos.startswith("JJ") and lem in self.cvb["AJ"]:
                if "V" in self.cvb["AJ"][lem]:
                    cands = self.cvb["AJ"][lem]["V"]
                    # best_cand = cands[0]
                    best_cand = max(cands, key=lambda x: len(x))
                    return [False, best_cand]

                elif "N" in self.cvb["AJ"][lem]:
                    cands = self.cvb["AJ"][lem]["N"]
                    # best_cand = cands[0]
                    best_cand = max(cands, key=lambda x: len(x))
                    return [False, best_cand]

                else:
                    return [False, lem]

            else:
                return [False, lem]

        elif pref2 == "il":
            reduced_form = lem[2:]
            return check_cvb_AJ(lem, reduced_form)

        elif pref3 == "irr":
            reduced_form = lem[2:]
            return check_cvb_AJ(lem, reduced_form)

        elif pref2 == "un":
            reduced_form = lem[2:]

            if lem[-3:] == "ful":
                return [True, lem[:-3]]

            if reduced_form in self.cvb["V"] and "V" in self.cvb["V"][reduced_form]:
                cands = self.cvb["V"][reduced_form]["V"]
                best_cand = cands[0]
                # best_cand = max(cands, key=lambda x: len(x))
                return [True, best_cand]

            elif reduced_form in self.cvb["AJ"]:
                if "V" in self.cvb["AJ"][reduced_form]:
                    cands = self.cvb["AJ"][reduced_form]["V"]
                    # best_cand = cands[0]
                    best_cand = max(cands, key=lambda x: len(x))
                    return [True, best_cand]

                elif "AJ" not in self.cvb["AJ"][reduced_form]:
                    return [True, reduced_form]

                elif "AJ" in self.cvb["AJ"][reduced_form]:
                    cands = self.cvb["AJ"][reduced_form]["AJ"]
                    best_cand = cands[0]
                    # best_cand = max(cands, key=lambda x: len(x))
                    return [True, best_cand]

                else:
                    return [False, lem]

            else:
                return [False, lem]

        elif pref2 == "im":

            if pos == "JJ" or pos.startswith("V"):

                reduced_form = lem[2:]

                if reduced_form in self.cvb["AJ"] and "AJ" in self.cvb["AJ"][reduced_form]:
                    return [True, self.cvb["AJ"][reduced_form]["AJ"][0]]

                elif reduced_form in self.cvb["AJ"] and "V" in self.cvb["AJ"][reduced_form]:
                    return [True, self.cvb["AJ"][reduced_form]["V"][0]]

                else:
                    return [False, lem]

            else:
                return [False, lem]

        elif pref2 == "in":
            if pos == "JJ":
                return check_cvb_AJ(lem, reduced_form=lem[2:])
            else:
                return [False, lem]

        # elif pref1 == "a":
        #     reduced_form = lem[1:]

        else:
            return [False, lem]  # lemma doesn't match negation pattern

    def check_negativity(self, token):

        # negative prefixes for English:
        # a–, dis–, il–, im–, in-, ir–, non–, un–

        lem = token["lemma"]

        if lem in ["no", "none", "never", "non"]:
            return True

        pos = token["pos"]

        pref3 = lem[:3]
        pref2 = lem[:2]
        pref1 = lem[0]

        post3 = lem[-3:]
        post2 = lem[-2:]

        reduced_form = reduced_form2 = None

        if pref3 == "irr":
            reduced_form = lem[2:]

        elif pref2 in ["il", "un", "im", "in"]:
            reduced_form = lem[2:]

        if reduced_form is not None:

            if pos == "JJ":
                if reduced_form in self.cvb["AJ"] or reduced_form[-4:] == "able":  # or reduced_form[:-2] in self.cvb["AJ"]:
                    return True

            elif pos == "RB" and reduced_form in self.cvb["AV"]:  # or reduced_form[:-2] in self.cvb["AV"]:
                return True

            elif pos in ["NN", "NNS"] and reduced_form in self.cvb["N"]:  # or reduced_form[:-2] in self.cvb["N"]:
                return True

            elif pos.startswith("V") and reduced_form in self.cvb["V"]:  # or reduced_form[:-2] in self.cvb["V"]:
                return True

            elif post3 == "nce" and pos.startswith("N"):
                return True

            elif (post2 == "ed" or post3 == "ous") and pos == "JJ":
                return True

        return False

    def is_JJR(self, word, prev_word, last_concept):

        if word[-2:] == "er":
            if last_concept not in self.cvb["AJ"]:
                print "JJR? ", word, last_concept

            return True

        elif prev_word in ["more", "less"] or last_concept == "more":
            return True

        else:
            return False

    def is_JJS(self, word, prev_word, last_concept):

        if word[-2:] == "st":
            if last_concept not in self.cvb["AJ"]:
                print "JJS? :", word, last_concept

            return True

        elif prev_word == "most" or last_concept == "most":
            return True

        else:
            return False

    def is_empty(self, word, pos, lem, dep):

        answer = False

        if word[0] in string.punctuation+"''":
            answer = True
            if len(word) > 1 and word[1].isdigit():
                answer = False

        elif pos in ["CC", "HYPH", "DT"]:
            answer = True

        elif word in ["a", "the", "an",
                      "is", "are", "was", "were", "be",

                      ]:

            answer = True

        return answer

    def reduce2verb(self, base, pos):

        cands = None
        pred = base

        if pos.startswith("N") and base in self.cvb["N"]:
            cands = self.cvb["N"][base].get("V", None)

        elif pos == "JJ" and base in self.cvb["AJ"]:
            cands = self.cvb["AJ"][base].get("V", None)

        elif pos == "RB" and base in self.cvb["AV"]:
            cands = self.cvb["AV"][base].get("V",None)

        if cands is not None:
            score = 0
            for c in cands:
                new_score = distance.get_jaro_distance(base, c, winkler=True)
                if new_score >= score:
                    score = new_score
                    pred = c

        return pred

    def tok2ctype(self, datum, amr, vconc_d, fhandle=None):

        tokens = datum["tokens"]

        for tokid, tok in enumerate(tokens):

            concepts = tok["concepts"]
            last_concept = concepts[-1]
            nform = self.get_tokbase(tok)
            word = tok["word"].lower()
            pos = tok["pos"]
            lem = tok["lemma"]
            dep = tok["rel"]

            # ================================================ #
            # CONSIDERING CONCEPT TYPES IN A PARTICULAR ORDER! #
            # ================================================ #

            # if len(concepts) > 2 and "name" not in concepts:
            #     print >> fhandle, "%s %s" % (word, concepts[1:])

            # not considering digits, as we can get concepts deterministically
            if word[0].isdigit():
                continue

            # skipping non-aligned concepts for now

            if lem in self.empties:
                tok["ctype"] = "empty"
                continue

            if last_concept is None:
                # # fixme 1: need treatment
                # if not (pos.startswith("N") or pos.startswith("V")):
                #     print >> fhandle, "%s" % word

                tok["ctype"] = "empty"
                continue

            last_nid = tok["node_id"][-1]

            if last_concept.startswith('"'):
                # find the gold type of NE
                self.fill_netype(amr, tok)

            if "-" in concepts:
                # fixme 1: need treatment
                # print >> fhandle, "%s" % word
                tok["neg"] = True

            if nform in self.main_dict:
                tok["ctype"] = "misc"

            elif pos == "JJR" and self.is_JJR(word, tokens[tokid-1]["word"].lower(), last_concept):
                tok["ctype"] = "comp_adj"

            elif pos == "JJS" and self.is_JJS(word, tokens[tokid-1]["word"].lower(), last_concept):
                tok["ctype"] = "sup_adj"

            elif last_concept.startswith('"'):
                if tok["netype"] == "person":
                    tok["ctype"] = "person"

                elif tok["netype"] == "organization":
                    tok["ctype"] = "org"

                elif tok["netype"] == "country":
                    tok["ctype"] = "country"

                else:
                    tok["ctype"] = "unk_ne"

            elif last_concept == lem[:-2] and lem[-2:] == "ly":
                # print >> fhandle, "%s %s" % (word, last_concept)#, [amr.node[x]["concept"] for x in pids])
                tok["ctype"] = "ly"

            elif last_concept == lem:
                tok["ctype"] = "lemma"

            elif last_concept == sng(word.lower(), pos="NOUN"):
                tok["ctype"] = "sng"

            elif self.patterns["verb"].match(last_concept):

                tok["ctype"] = "verb"
                items = last_concept.split("-")

                if len(items) != 2:
                    # cannot handle: have-concession-91, have-condition-91, etc.
                    tok["ctype"] = "empty"
                    # print >> fhandle, "%s %s" % (word, last_concept)
                    continue

                elif "person" in concepts:
                    tok["ctype"] = "person-verb"

                elif "thing" in concepts:
                    tok["ctype"] = "thing-verb"

                base = self.get_verbbase(lem.lower(), pos)
                pref = base[:2]
                if pref not in vconc_d:
                    # only one pair: if -> cause-01
                    tok["ctype"] = "empty"
                    continue

                # prediction = self.get_vconcept(base, pos, pref, vconc_d)
                prediction = self.get_vconcept_dummy(base, pos, pref, vconc_d)

                # fixme change to the line below (to go back to verb-num1num2)
                if prediction[:-3] != last_concept[:-3]:
                # if prediction[:-3] != last_concept[:-3] and last_concept[-2:] != "00":
                    # fixme: we need to assign "empty"
                    # because these are difficult to manage
                    tok["ctype"] = "empty"
                    # print >> fhandle, "%s %s" % (base, last_concept)

            elif lem in self.cvb["AJ"] and "N" in self.cvb["AJ"][lem] and last_concept == self.cvb["AJ"][lem]["N"][0]:
                # print "JJ2N: %s %s %s" % (tok["word"].lower(), last_concept, concepts)
                tok["ctype"] = "adj2n"

            elif lem in self.cvb["AV"] and "N" in self.cvb["AV"][lem] and last_concept == self.cvb["AV"][lem]["N"][0]:
                # print "AV2N: %s %s %s" % (tok["word"].lower(), last_concept, concepts)
                tok["ctype"] = "adv2n"

            else:
                tok["ctype"] = "empty"
                # print >> fhandle, "%s %s %s" % (word, pos, last_concept)

    # =============================================== CTYPE --> CONCEPT

    def fill_more_less_concepts(self, G, tokens, token, tokid, word):

        concept = word

        if tokid <= len(tokens) - 3:
            next_word = tokens[tokid + 1]["word"].lower()
            next_word2 = tokens[tokid + 2]["word"].lower()

            if next_word == "and" and next_word2 == word:
                concept = "%s-and-%s" % (word, word)

        elif tokid <= len(tokens) - 2 and tokens[tokid + 1]["word"].lower() == "than":
            concept = "%s-than" % word

        self.fill_one_node(G, tokid, concept, var=True)

    def fill_misc_concepts(self, G, tokid, token, misc_concepts):

        concepts_type = type(misc_concepts)

        if concepts_type == str:
            var = False if misc_concepts == "-" else True
            self.fill_one_node(G,tokid, misc_concepts, var)

        elif concepts_type == tuple:
            hc, tc, role = misc_concepts
            hvar = False if hc == "-" else True
            tvar = False if tc == "-" else True
            self.fill_triple(G, tokid, hc, hvar,tc, tvar,role)

        elif concepts_type == dict:
            for k,v in misc_concepts.items():
                self.fill_one_node(G, tokid, k, var=True)
                hid = G.graph["maxid"]
                for pair in v:
                    tc, role = pair
                    tvar = False if tc == "-" else True
                    self.fill_one_node(G, tokid, tc, var=tvar)
                    G.add_edge(hid, G.graph["maxid"], {"label": role})

        else:
            logger.info("Unknown type of misc concepts: %s" %(misc_concepts))
            sys.exit()

    def fill_one_node(self, G, tokid, concept, var=True):

        G.graph["maxid"] += 1

        # fix concepts with a slash (as they usually break smatch scoring)
        if "/" in concept and '"' not in concept:
            concept = '"%s"' % concept

        # create a dummy concept for the RI stage
        if re.match(self.patterns["verb"], concept) is None:
            dummy_concept = concept

        else:
            parts = concept.split("-")[:-1]
            dummy_concept = "%s-00" %("-".join(parts))

        if var:

            G.graph["varnum"] += 1
            G.add_node(G.graph["maxid"], {"concept": dummy_concept, "var": "x%d" % G.graph["varnum"],
                                          "true_concept": concept,
                                          "isroot": False, "isleaf": True,
                                          "span": (tokid, tokid + 1), "aux_span": None,
                                          "suc": [], "pred": []})

        else:
            G.add_node(G.graph["maxid"], {"concept": dummy_concept, "var": None,
                                          "true_concept": concept,
                                          "isroot": False, "isleaf": True,
                                          "span": (tokid, tokid + 1), "aux_span": None,
                                          "suc": [], "pred": []})

    def fill_triple(self,  G, tokid, hc, hvar, tc, tvar, role):

        self.fill_one_node(G, tokid, hc, var=hvar)
        nid1 = G.graph["maxid"]

        self.fill_one_node(G, tokid, tc, var=tvar)
        G.add_edge(nid1, G.graph["maxid"], {"label": role})

    def fill_date_interval(self, G, tokid, parts):

        di = "date-interval"
        dc = "date-entity"
        y1, y2 = parts

        self.fill_one_node(G, tokid, di, var=True)
        date_interval_nid = G.graph["maxid"]

        self.fill_one_node(G, tokid, dc, var=True)
        date_entity_nid1 = G.graph["maxid"]

        self.fill_one_node(G, tokid, dc, var=True)
        date_entity_nid2 = G.graph["maxid"]

        self.fill_one_node(G, tokid, "%s" % y1, var=False)
        year1_nid = G.graph["maxid"]

        self.fill_one_node(G, tokid, "%s" % y2, var=False)
        year2_nid = G.graph["maxid"]

        G.add_edge(date_interval_nid, date_entity_nid1, {"label": "op"})
        G.add_edge(date_interval_nid, date_entity_nid2, {"label": "op"})
        G.add_edge(date_entity_nid1, year1_nid, {"label": "year"})
        G.add_edge(date_entity_nid2, year2_nid, {"label": "year"})

    # =============

    def get_vconcept(self, word, pos, pref, vidx):

        word = self.reduce2verb(word, pos)  # trying to look-up a token in CVB
        vconcept = "%s-01" % word
        cands = vidx[pref]
        score = 0

        for c in cands:
            # each candidate is of the form verb-num1num2
            # so we strip the last 3 symbols
            new_score = distance.get_jaro_distance(word, c[:-3], winkler=True)
            if new_score >= score:
                score = new_score
                vconcept = c

        return vconcept

    def get_vconcept_dummy(self, word, pos, pref, vidx):

        word = self.reduce2verb(word, pos)  # trying to look-up a token in CVB
        vconcept = word
        cands = vidx[pref]
        score = 0

        for c in cands:
            # each candidate is of the form verb-num1num2
            # so we strip the last 3 symbols
            verb_cand = c[:-3]
            new_score = distance.get_jaro_distance(word, verb_cand, winkler=True)
            if new_score >= score:
                score = new_score
                vconcept = verb_cand

        return "%s-00" % vconcept

    def get_concept_from_adj(self, adj):

        # if "V" in self.cvb["AJ"][adj]: # fixme I don't have to consider this case as it is handled in the precious pass
        #     c = "%s-01" % self.cvb["AJ"][adj]["V"][0]

        if "N" in self.cvb["AJ"][adj]:
            c = self.cvb["AJ"][adj]["N"][0]

        else:
            c = adj

        return adj

    def get_concept_from_ly(self, token):

        # If tok["lemma"][-2:] == "ly" and "pos" in ["JJ", "RB"]
        # we need to determine whether the concept is formed according to some morph. derivation pattern

        lem = token["lemma"].lower()

        # 1. an AV formed by adding -ly to an adjective
        op1 = lem[:-2]

        # 2. if AJ ends with "-y", we can create an AV
        # by replacing the "-y" with "i" and adding "ly"
        op2 = "%sy" % lem[:-3]

        # 3. if AJ ends with "-able", "-ible", or "-le", we can create an AV
        # by replacing the "-e" with "-y"
        op3 = "%se" % lem[:-1]

        # 4. if AJ ends with "-ic" (except "public"),
        # we can crete an AV by adding "-ally"
        op4 = lem[:-4]

        for option in [op1, op2, op3, op4]:

            if option in self.cvb["AJ"]:
                # return self.get_concept_from_adj(option)
                return option
            else:
                return lem

    def fill_comp_adjc(self, G, tokens, tokid, lem):

        prev_word = tokens[tokid - 1]["word"].lower() if tokid >= 1 else None
        degree = "more"

        if prev_word in ["less", "more"]:
            degree = prev_word
            concept = lem

        else:
            concept = lem[:-2]

        self.fill_one_node(G, tokid, concept, var=True)
        nid1 = G.graph["maxid"]

        self.fill_one_node(G, tokid, degree, var=True)
        G.add_edge(nid1, G.graph["maxid"], {"label": "degree"})

    def fill_sup_adjc(self, G, tokid, lem):

        if lem[-2:] == "st":
            concept = lem[:-3]

        else:
            concept = lem

        self.fill_one_node(G, tokid, concept, var=True)
        nid1 = G.graph["maxid"]

        self.fill_one_node(G, tokid, "most", var=True)
        G.add_edge(nid1, G.graph["maxid"], {"label": "degree"})

    def fill_url_concept(self, G, tokid, word):

        hc = "url-entity"
        hvar = True

        tc = '"%s"' % word
        tvar = False

        role = "value"
        self.fill_triple(G, tokid, hc, hvar, tc, tvar, role)

    def fill_digit_ctype(self,G, tokens, tokid, token, word, pos):

        snt_len = len(tokens)

        if "-" in word:

            parts = word.split("-")

            # case 1
            if len(parts) == 3:
                if date_pat.match(word) is not None:
                    self.fill_date_concept(G, tokid, token)

                else:
                    print "UNK -DATE PATTERN: %s" % word

            elif token["ne"] in ["DATE", "DURATION"] and re.match("(1|2)\d{3}-(1|2)\d{3}", word) is not None:
                self.fill_date_interval(G, tokid, parts)

            elif len(parts) > 3:
                print "UNKNOWN -DIGIT PATTERN: %s" %word
                self.fill_one_node(G, tokid, word, var=False)

            else:
                print "UNKNOWN -DIGIT PATTERN: %s" % word
                for i in parts:
                    self.fill_one_node(G, tokid, i, var=False)

        elif "/" in word:
            parts = word.split("/")

            # 9/13/2000
            if len(parts) == 3:
                self.fill_date_concept(G, tokid, token)

            # 1/4th
            elif len(word) ==5 and word[-2:] in ["st", "nd", "rd", "th"]:
                self.fill_one_node(G, tokid, '"%s"' % word[:-2], var=False)

            # 1/4
            else:
                self.fill_one_node(G, tokid, '"%s"' %word, var=False)

        elif pos == "LS" and tokid == 0:
            # print "ls:", word, concepts, " ".join([T["word"] for T in tokens])
            # fixme deterministic edge :li
            self.fill_one_node(G, tokid, word, var=False)

        elif re.match("\d+(st|nd|rd|th)", word) is not None:

            if "century" in [T["word"].lower() for T in tokens[tokid:tokid + 3]]:
                # print "century date-entity: ", word, concepts, " ".join([T["word"] for T in tokens])
                self.fill_one_node(G, tokid, "date-entity", var=True)
                date_entity_nid = G.graph["maxid"]

                self.fill_one_node(G, tokid, word[:-2], var=False)
                G.add_edge(date_entity_nid, G.graph["maxid"], {"label": "century"})

            else:
                # print "ordinal: ", word, concepts, " ".join([T["word"] for T in tokens])
                self.fill_one_node(G, tokid, "ordinal-entity", var=True)
                hid = G.graph["maxid"]

                self.fill_one_node(G, tokid, word[:-2], var=False)
                G.add_edge(hid, G.graph["maxid"], {"label": "value"})

        elif word[-1] == "s":
            self.fill_one_node(G, tokid, "date-entity", var=True)
            hid = G.graph["maxid"]

            self.fill_one_node(G, tokid, word[:-1], var=False)
            G.add_edge(hid, G.graph["maxid"], {"label": "decade"})

        elif re.match(r"^\d{4}$", word):

            # prev2 = tokens[tokid-2]["word"] if tokid >= 2 else " "
            prev1 = tokens[tokid - 1]["word"] if tokid >= 1 else " "

            # next1 = tokens[tokid +1]["word"] if tokid <= snt_len - 2 else " "
            # next2 = tokens[tokid +2]["word"] if tokid <= snt_len - 3 else " "

            if prev1 in ["in", "from", "to", "by", "for"] and token["ne"] == "DATE":

                self.fill_one_node(G, tokid, "date-entity", var=True)
                hid = G.graph["maxid"]

                self.fill_one_node(G, tokid, word, var=False)
                G.add_edge(hid, G.graph["maxid"], {"label": "year"})
                # print prev2, prev1, word, next1, next2, token["ne"]

            else:
                self.fill_one_node(G, tokid, word, var=False)

        elif ":" in word:
            # # TODO deal later
            # if tokid >=1 and tokens[tokid-1]["word"].lower() in ["at","about","around"]:
            #     # print "century date-entity: ", word, concepts, " ".join([T["word"] for T in tokens])
            #     self.fill_one_node(G, tokid, "date-entity", var=True)
            #     date_entity_nid = G.graph["maxid"]
            #
            #     self.fill_one_node(G, tokid, word, var=False)
            #     G.add_edge(date_entity_nid, G.graph["maxid"], {"label": "century"})

            # ad-hoc solution
            self.fill_one_node(G, tokid, '"%s"' %word, var=False)

        else:
            self.fill_one_node(G, tokid, word, var=False)

    def fill_date_concept(self, G, tokid, token):

        word = token["word"]
        date_elems = token["word"].split("-")

        try:
            date = dateutil.parser.parse(word, ignoretz=True).strftime("%Y/%-m/%-d").split("/")
            self.fill_one_node(G, tokid, "date-entity", var=True)  # main node in the fragment
            date_entity_nid = G.graph["maxid"]
            for item, edge_label in zip(date, ["year", "month", "day"]):  # date = [year, month, day]
                self.fill_one_node(G, tokid, item, var=False)
                G.add_edge(date_entity_nid, G.graph["maxid"], {"label": edge_label})

        except ValueError:
            # print "VALUE ERROR: ", word

            # one of the elements is zero (2000-00-03 or 0000-04-12)
            assert len(date_elems) == 3  # [year, month, day]
            self.fill_one_node(G, tokid, "date-entity", var=True)
            date_entity_nid = G.graph["maxid"]
            year, month, day = date_elems

            if year != "0000":
                self.fill_one_node(G, tokid, year, var=False)
                G.add_edge(date_entity_nid, G.graph["maxid"], {"label": "year"})

            if month != "00":
                if month.startswith("0"):
                    month = re.sub("0", "", month)
                self.fill_one_node(G, tokid, month, var=False)
                G.add_edge(date_entity_nid, G.graph["maxid"], {"label": "month"})

            if day != "00":
                if day.startswith("0"):
                    day = re.sub("0", "", day)
                self.fill_one_node(G, tokid, day, var=False)
                G.add_edge(date_entity_nid, G.graph["maxid"], {"label": "day"})

        except TypeError:
            print "Type error!", word

        except OverflowError:
            print "Overflow error!", word

    def fill_ne_concept(self, G, tokid, word, ne_type):

        if ne_type == "unk_ne":
            # self.fill_one_node(G, tokid, "unk_ne", var=True)
            # ne_type_nid = G.graph["maxid"]

            self.fill_one_node(G, tokid, "name", var=True)
            name_nid = G.graph["maxid"]
            # G.add_edge(ne_type_nid, name_nid, {"label": "name"})

            self.fill_one_node(G, tokid, '"%s"' % word, var=False)
            G.add_edge(name_nid, G.graph["maxid"], {"label": "op"})

        else:

            # need to check if there are nodes of the same type in G
            ne_type_nids = filter(lambda x: G.node[x]["concept"] == ne_type, G.nodes(data=False))
            if len(ne_type_nids) > 0:
                for nid in ne_type_nids:
                    if G.node[nid]["span"][1] == tokid and len(G.successors(nid)) > 0:
                        name_nid = G.successors(nid)[0]
                        # assert G.node[name_nid]["concept"] == "name"
                        self.fill_one_node(G, tokid, '"%s"' % word, var=False)
                        G.add_edge(name_nid, G.graph["maxid"], {"label": "op"})
                        return

            self.fill_one_node(G, tokid, ne_type, var=True)
            ne_type_nid = G.graph["maxid"]

            self.fill_one_node(G, tokid, "name", var=True)
            name_nid = G.graph["maxid"]
            G.add_edge(ne_type_nid, name_nid, {"label": "name"})

            self.fill_one_node(G, tokid, '"%s"' % word, var=False)
            G.add_edge(name_nid, G.graph["maxid"], {"label": "op"})

    def fill_verbc(self, lem, pos, vidx, G, tokid):

        base = self.get_verbbase(lem, pos)
        pref = base[:2]

        if pref not in vidx:
            vconcept = "%s-01" %lem
        else:
            vconcept = self.get_vconcept(base, pos, pref, vidx)

        self.fill_one_node(G, tokid, vconcept, var=True)

    def fill_concepts_from_ctypes(self, tokens, tokid, token, cg, G, vidx):

        lem = token["lemma"]
        pos = token["pos"]

        if cg == "empty":
            return

        elif cg == "verb":
            self.fill_verbc(lem, pos, vidx, G, tokid)

        elif cg in ["person", "country", "unk_ne"]:
            self.fill_ne_concept(G, tokid, token["word"], cg)

        elif cg == "org":
            self.fill_ne_concept(G, tokid, token["word"], "organization")

        elif cg == "sng":
            concept = sng(token["word"].lower(), pos="NOUN")
            self.fill_one_node(G, tokid, concept, var=True)

        elif cg == "lemma":
            self.fill_one_node(G, tokid, token["lemma"], var=True)

        elif cg == "ly":
            # self.fill_one_node(G, tokid, token["lemma"][:-2], var=True) # fixme performs better?
            concept = self.get_concept_from_ly(token)
            self.fill_one_node(G, tokid, concept, var=True)

        elif cg == "comp_adj":
            self.fill_comp_adjc(G, tokens, tokid, lem)

        elif cg == "sup_adj":
            self.fill_sup_adjc(G, tokid, lem)

        elif cg == "adj2n":
            try:
                concept = self.cvb["AJ"][lem]["N"][0]
            except KeyError:
                concept = lem
            self.fill_one_node(G, tokid, concept, var=True)

        elif cg == "adv2n":
            try:
                concept = self.cvb["AV"][lem]["N"][0]
            except KeyError:
                concept = lem
            self.fill_one_node(G, tokid, concept, var=True)

        elif cg == "thing-verb":

            self.fill_one_node(G, tokid, "thing", var=True)
            nid1 = G.graph["maxid"]

            self.fill_verbc(lem, pos, vidx, G, tokid)
            G.add_edge(nid1, G.graph["maxid"], {"label": "ARG0-of"})

        elif cg == "person-verb":
            self.fill_one_node(G, tokid, "person", var=True)
            nid1 = G.graph["maxid"]

            self.fill_verbc(lem, pos, vidx, G, tokid)
            G.add_edge(nid1, G.graph["maxid"], {"label": "ARG0-of"})

    def get_legal_ctype(self, token):

        word = token["word"]

        ne = ["org", "person", "country", "unk_ne"]

        if len(word) <=2:
            return ["empty", "lemma", "verb"] + ne

        else:
            return self.ctypes


if __name__ == "__main__":
    p = "cats"
    x = "understanding"
    print "plural: %s, singular: %s, %s" % (p, sng(p), sng(x))
