#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import logging
import os,sys
import re
import networkx as nx

import lib.rules
from base_reader import FileReader

logger = logging.getLogger("main")
ruleset = lib.rules.Ruleset()
ruleset.load_db()

class JamrFileReader(FileReader):

    def get_comments(self, id_line, snt_line, save_line, tok_line):
        keys = []
        values = []
        for cmt in [id_line, snt_line, save_line, tok_line]:
            elems = cmt.split("::")
            for el in elems[1:]:  # do not need the '# ' part
                pair = el.split()
                if len(pair) > 1:
                    keys.append(pair[0])
                    values.append(" ".join(pair[1:]))
                else:
                    keys.append(pair[0])
                    values.append("True")

        return dict(zip(keys, values))


    def gen_tok_file(self, fname, data=None):

        """
        1) Process an AMR file (fname) and create 3 files:
        fname.amr.tok - AMR with a "# ::tok" field (input for JAMR Aligner)
        fname.amr.tok.amr - (input for ISI Aligner)
        fname.amr.tok.eng - (input for ISI Aligner)

        2) Populate each instance's "comments" dictionary

        :param fname: the name of the input file
        :param data: a list of data dictionaries

        """

        path = "%s.amr.tok" % fname
        amr_tokfile = open(path, "w")
        ISI_eng_file = open("%s.eng" % path, "w")
        ISI_amr_file = open("%s.amr" % path, "w")

        JAMR_amr_string = []  # native PENMAN notation
        JAMR_amr_list = []
        ISI_amr_string = []  # one-line amr string
        ISI_amr_list = []

        id_pat = re.compile(r"^# ::id.*")
        snt_pat = re.compile(r"^# ::snt.*")
        save_pat = re.compile(r"^# ::save-date.*")

        id_list = []
        snt_list = []
        save_list = []

        inst = False

        for line in open(fname):
            if line.startswith("#"):
                if id_pat.match(line):
                    id = line.strip()
                    id_list.append(id)
                    inst = True
                elif snt_pat.match(line):
                    snt = line.strip()
                    snt_list.append(snt)
                elif save_pat.match(line):
                    save = line.strip()
                    save_list.append(save)

            elif not line.strip() and len(JAMR_amr_string) > 0:
                JAMR_amr_list.append("".join(JAMR_amr_string))
                ISI_amr_list.append(" ".join(ISI_amr_string))
                JAMR_amr_string = []
                ISI_amr_string = []
                inst = False

            elif len(line) > 1:
                inst = True
                JAMR_amr_string.append(line)
                ISI_amr_string.append(line.strip())

        if inst:
            JAMR_amr_list.append("".join(JAMR_amr_string))
            ISI_amr_list.append(" ".join(ISI_amr_string))

        idl, sntl, savel, ISIal, oal = len(id_list), \
                                       len(snt_list), \
                                       len(save_list), \
                                       len(ISI_amr_list), \
                                       len(JAMR_amr_list)

        assert idl == sntl == savel == ISIal == oal > 0, \
            "Mismatch: %d id_cmt, %d snt_cmt, %d save_cmt, %d ISI amrs, %d JAMR amrs" \
            % (idl, sntl, savel, ISIal, oal)

        for i, datum in enumerate(data):
            snt = " ".join(tok["word"] for tok in datum["tokens"])
            tok_snt = "# ::tok %s" % (snt)
            id = id_list[i]

            # storing comments in an instance
            datum["comments"] = self.get_comments(id, snt_list[i], save_list[i], tok_snt)

            # writing to an amr.tok file - for the JAMR aligner
            amr_tokfile.write("%s\n%s\n%s\n" % (id, tok_snt, JAMR_amr_list[i]))

            # writing to an .eng and .amr files - for the ISI aligner
            ISI_eng_file.write("%s\n" % (snt))
            ISI_amr_file.write("%s\n" % (ISI_amr_list[i]))

        amr_tokfile.close()
        ISI_amr_file.close()
        ISI_eng_file.close()

    def get_dummy_concept(self, concept):
        parts = concept.split("-")
        if len(parts) == 2:
            dummy_concept = "%s-00" % parts[0]
        else:
            dummy_concept = "%s-00" % ("-".join(parts[:-1]))

        return concept, dummy_concept

    def process_nodeline(self, line, verb_pat, datum):

        elems = line.strip().split()[2:]
        node_id = elems[0]
        start_tok_id = end_tok_id = None

        if len(elems) > 2:  # (node_id \t concept \t span or node_id \t "concept_1 concept_2"
            if elems[-1][-1].isdigit():  # aligned!  (0.4.0.0 \t pack \t 1-2)
                concept = " ".join(elems[1:-1])

                if verb_pat.match(concept):
                    true_concept, dummy_concept = self.get_dummy_concept(concept)
                else:
                    true_concept = dummy_concept = concept

                # this is done for better training of the RI
                start_tok_id, end_tok_id = map(lambda x: int(x), elems[-1].split("-"))

                token_range = end_tok_id - start_tok_id

                if token_range == 1:
                    datum["tokens"][start_tok_id]["concepts"] += [dummy_concept]
                    datum["tokens"][start_tok_id]["node_id"].append(node_id)
                    self.stats_fhandle.write("%s %s\n" % (datum["tokens"][start_tok_id]["word"], concept))

                elif token_range >= 2:
                    for i in range(start_tok_id, end_tok_id):
                        datum["tokens"][i]["concepts"] += [dummy_concept]
                        datum["tokens"][i]["node_id"].append(node_id)

                    # if (end_tok_id - start_tok_id) >= 2:
                    #     tokids = range(start_tok_id, end_tok_id)
                    #     tokens = [datum["tokens"][i] for i in tokids]
                    #     self.stats_fhandle.write("%s %s\n" % (" ".join([T["word"] for T in tokens]), concept))

                else:
                    start_tok_id = end_tok_id = None

            else:  # not aligned !
                concept = " ".join(elems[1:])

        else:  # not aligned !
            concept = elems[-1]

        # fixme: comment out if want to go back to normal verb-num1num2 concepts
        if verb_pat.match(concept):
            true_concept, dummy_concept = self.get_dummy_concept(concept)
        else:
            true_concept = dummy_concept = concept

        return node_id, true_concept, dummy_concept, start_tok_id, end_tok_id

    def process_edgeline(self, line, amr, edges):
        elems = line.strip().split()[2:]
        relation, parent_id, child_id = elems[1], elems[-2], elems[-1]
        amr.node[parent_id]["isleaf"] = False
        amr.node[parent_id]["suc"].append(child_id)
        amr.node[child_id]["pred"].append(parent_id)

        # normalizing "op_n" relation label
        if re.match(r"op\d+", relation):
            relation = "op"

        if re.match(r"snt\d+", relation):
            relation = "snt"

        edge = (parent_id, child_id, {"label": relation})

        amr.add_edge(*edge)
        edges[relation] = edges.get(relation,0) + 1

    def process_amrline(self, line, amr, var_pat, verb_pat, this_aux_c2n):

        m = var_pat.finditer(line)
        if m is not None and not line.startswith("#"):
            for x in m:
                variable, conc = x.group(3), x.group(5)

                if verb_pat.match(conc):
                    parts = conc.split("-")
                    if len(parts) == 2:
                        conc = "%s-00" % parts[0]
                    else:
                        conc = "%s-00" % ("-".join(parts[:-1]))

                N_id = this_aux_c2n[conc].pop()
                amr.node[N_id]["var"] = variable

    def init_node(self, dummy_concept, true_concept, sid, eid):

        # note that we store dummy_concept as the main one - for RI training
        return {"concept": dummy_concept, "true_concept": true_concept,
                "elabel": None, "var": None,
                "isroot": False, "isleaf": True,
                "span": (sid, eid), "aux_span": None,
                "suc": [], "pred": [],

                }

    def read_file(self):

        logger.info("Reading aligned AMR file")
        self.stats_fhandle = open(os.path.abspath(os.path.join(self.params["stat_dir"], "JAMR_prp_stats.txt")), "w")

        var_pat = lib.rules.var_pat
        verb_pat = lib.rules.verb_pat
        vidx = self.vocab["vidx"]

        edges = {}
        # ======================================================= MAIN ============================================
        # current variables
        this_idx = 0
        this_datum = self.data[this_idx]
        this_amr = nx.DiGraph(id=this_datum["comments"]["id"])
        this_aux_c2n = {}

        for line in open(self.fname):

            if line[:8] == "# ::node":
                nid, true_con, dummy_con, start_tid, end_tid = self.process_nodeline(line, verb_pat, this_datum)
                na = self.init_node(dummy_con, true_con, start_tid, end_tid)
                this_amr.add_node(nid, attr_dict=na)

                # keep track of node concept <--> id mapping (need for var --> concept map)
                this_aux_c2n.setdefault(dummy_con, []).append((nid))

            elif line[:8] == "# ::root":
                this_amr.node[line.strip().split()[2]]["isroot"] = True

            elif line[:8] == "# ::edge":
                self.process_edgeline(line, this_amr,edges)

            # ===================================== 4. FINAL ADJUSTMENTS ======================================

            elif not line.strip() and len(this_aux_c2n.keys()) > 0:
                ruleset.tok2ctype(this_datum, this_amr, vidx)
                this_datum["amrG"] = this_amr

                # move to the next instance
                # logger.info("Processed instance %d" % (this_idx + 1))
                this_idx += 1
                try:
                    this_datum = self.data[this_idx]
                    this_amr = nx.DiGraph(id=this_datum["comments"]["id"])
                    this_aux_c2n = {}
                    continue

                except IndexError:
                    break

            # =================================== 3. GET THE VARIABLE MAPPING =============================================

            else:
                # we are finished with the alignments
                # populate variable-to-concept mapping
                self.process_amrline(line, this_amr, var_pat, verb_pat, this_aux_c2n)
                continue

        assert this_idx == len(self.data)
        self.stats_fhandle.close()

        if "edges" not in self.vocab:
            self.vocab["ctypes"] = lib.rules.ctypes
            self.vocab["edges"] = edges


class ISIReader(FileReader):

    def read_file(self):

        """
        Read ISI alignments (fname) and get:
        - concept-to-token mapping
        - a string representation for an amr graph
        """

        L = []
        with open(self.fname, "r") as alf:
            for line in alf:
                if line.startswith("#"):
                    continue
                elif not line.strip():
                    continue
                else:
                    con2tok = collections.defaultdict(collections.deque)
                    var2conc = collections.defaultdict(str)

                    # delete all brackets
                    y = re.sub(r"\(|\)", "", line.strip())
                    # substitute roles for @
                    y = re.sub(r":[a-zA-Z].*?\s", "@", y)
                    elems = [el for el in y.split("@")]

                    for el in elems:
                        var_conc_pair = el.strip().split(" / ")

                        if len(var_conc_pair) == 1:
                            if var_conc_pair in var2conc.keys():  # it is a variable, don't need it
                                continue
                            else:
                                components = var_conc_pair[0].split("~e.")
                                if len(components) > 1:  # aligned!
                                    C, T = components
                                    con2tok[C].append(int(T))

                                else:  # not aligned !
                                    C, T = components[0], None
                                    con2tok[C].append(None)

                        else:  # we have a [var, concept] pair
                            var, concept = var_conc_pair
                            var2conc[var] = concept
                            components = concept.split("~e.")

                            if len(components) > 1:  # alinged!
                                C, T = components
                                con2tok[C].append(int(T))

                            else:  # not aligned !
                                C, T = components[0], None
                                con2tok[C].append(None)

                    L.append((con2tok, line.strip()))

        return L