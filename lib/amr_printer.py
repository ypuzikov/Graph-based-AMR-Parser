#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import random
import logging
import networkx as nx
import pdb
import cPickle as pickle

logger = logging.getLogger("main")

class Printer(object):
    def __init__(self):
        pass

    def save_penman(self, graphs, amr_fn, edges_fn, true_concepts=True):

        with open(amr_fn, "w") as amr_out:
            edge_out = open(edges_fn, "w")

            for g in graphs:
                amrstring = self.amr_to_string(g) if true_concepts else self.amr_to_string(g, true_concepts=False)
                comments = g.graph["comments"]
                amr_out.write("# ::id %s ::annotator PAMR\n" % comments["id"])
                amr_out.write("# ::snt %s\n" % comments["snt"])
                amr_out.write("# ::tok %s\n" % comments["tok"])
                amr_out.write("%s\n\n" % amrstring)

                edgestrings = "\n".join([self.gen_edge_string(g,e) for e in g.edges(data=True)])
                edge_out.write("%s\n%s\n" % (comments["id"], edgestrings))

            edge_out.close()

            logger.info("Predicted AMR (PENMAN) saved to: \n%s" % (amr_fn))
            logger.info("Predicted edges saved to: \n%s " % (edges_fn))

    def amr_to_string(self, G, true_concepts=True):

        comments = G.graph["comments"]
        number_of_children_to_process = []
        amr_string = []
        varlist = []
        counters = {"op": 1, "snt": 1}

        if true_concepts:
            for nid, na in G.nodes(data=True):
                na["concept"] = na["true_concept"]

        if len(G.edges()) == 0:
            conceptstr = " ".join([G.node[nid]["concept"] for nid in G.nodes()])
            print "\nNo edges in G (%s). Concepts: %s" % (comments["id"], conceptstr)
            random_concept = G.node[G.nodes()[0]]["concept"]
            return "(%s / %s)" % ("x", random_concept)

        try:
            nodes_order = nx.topological_sort(G)
        except nx.exception.NetworkXUnfeasible:
            nodes_order = self.fix_cycle(G)

        root_node = nodes_order[0]
        self.get_node_string(G, root_node, amr_string, number_of_children_to_process, varlist, counters)

        try:
            return " ".join(amr_string)
        except TypeError:
            print ("\nType error when returning amr string (%s):\n%s" % (comments["id"], amr_string))
            random_concept = G.node[G.nodes()[0]]["concept"]
            return "(%s / %s)" % ("x", random_concept)

    def get_node_string(self, amr, u, amr_string, num_of_children_to_process, visited_vars, counters):

        """
        An aux function to populate the amr_string list.
        Checks whether a node is ready to be closed with a bracket,
        depending on the number of the remaining children nodes to be processed

        """

        # first determine if there are multiple parents (re-entrance)
        par_num = len(amr.predecessors(u))
        if par_num > 1:
            self.fix_multi_parents(u, amr, visited_vars)

        # check if current node "u" is a leaf node
        ch_num = len(amr[u])
        if ch_num == 0:
            # current node is a leaf,
            # either a variable-bearing one or a Constant
            self.process_leafnode(amr, u, amr_string, visited_vars, num_of_children_to_process)

        else:

            # current node is not a leaf => has children
            # fixme here is the part which needs to be fixed: we have false assignment of children to constants
            var = amr.node[u]["var"]
            concept = amr.node[u]["concept"]

            try:
                node_string = "(%s" % (" / ".join([var, concept]))
            except TypeError:
                # this happens because a non-variable node was assigned children
                # e.g. (4, fork, ARG2)
                print("False variable assignment: concept %s (%s)" % (concept, amr.graph["comments"]["id"]))
                node_string = "(%s" % (" / ".join(["x%d" % (random.randint(0, 999999)), concept]))

            amr_string.append(node_string)
            visited_vars.append(var)
            num_of_children_to_process.append(ch_num)

            sorted_ch = sorted(amr[u], key=lambda x: amr.node[x]["var"] is not None, reverse=True)
            for v in sorted_ch:
                # process each child in turn

                edge = amr[u][v]["label"]
                if edge == "op":
                    edge = "op%d" % (counters["op"])
                    counters["op"] += 1
                elif edge == "snt":
                    edge = "snt%d" % (counters["snt"])
                    counters["snt"] += 1

                amr_string.append(":%s" % (edge))

                # if we find a variable, we need to skip it,
                # because we have already looked at the referred node once.
                if amr.node[v]["var"] in visited_vars:
                    amr_string.append(amr.node[v]["var"])

                    # Still, we need to adjust the children-to-process count accordingly
                    num_of_children_to_process[-1] -= 1
                    self.ask_prev_parents(num_of_children_to_process, amr_string)

                else:
                    self.get_node_string(amr, v, amr_string, num_of_children_to_process, visited_vars, counters)

    def fix_cycle(self, G):
        logger.debug("G is cyclic -> adjusting edge labels ...")

        cycles = list(nx.simple_cycles(G))
        for c in cycles:
            if len(c) == 1:
                logger.debug("A one-node cycle!")
                hid = c.pop(0)
                G.remove_edge(hid, hid)
                continue

            hid = c.pop(0)
            tid = c.pop(0)
            label = G[hid][tid]["label"]

            G.remove_edge(hid, tid)

            lab_parts = label.split("-")

            if lab_parts[-1] == "of":
                G.add_edge(tid, hid, {"label": "%s" % (lab_parts[0])})
            else:
                G.add_edge(tid, hid, {"label": "%s-of" % (label)})

        return nx.topological_sort(G)

    def fix_multi_parents(self, curr_node, graph, visited_variables):
        # we filter out processed parent nodes
        pids = [pid for pid in graph.predecessors(curr_node) if graph.node[pid]["var"] not in visited_variables]
        parent_num = len(pids)
        while parent_num > 0:
            # edges from all not-visited parents should be reified
            pid = pids.pop()
            rel_lab = graph[pid][curr_node]["label"]
            lab_parts = rel_lab.split("-")
            # (pid, u, ARG1-of) -> (u, pid, ARG1)
            if lab_parts[-1] == "of":
                graph.add_edge(curr_node, pid, {"label": "%s" % (lab_parts[0])})
            # (pid, u, ARG1) -> (u, pid, ARG1-of)
            else:
                graph.add_edge(curr_node, pid, {"label": "%s-of" % (rel_lab)})

            # remove (pid, u) edge, because we just added a new one with the opposite direction
            graph.remove_edge(pid, curr_node)
            pids = [pid for pid in graph.predecessors(curr_node) if graph.node[pid]["var"] not in visited_variables]
            parent_num = len(pids)

        return

    def ask_prev_parents(self, number_of_children_to_process, amr_string):
        # ask previous parent nodes whether they have more children to be processed
        while True:
            prev_ch = number_of_children_to_process[-1]
            if prev_ch > 0:
                # if so, don't close them with a bracket
                break

            # else, add a bracket to close the parent,
            # pop the parent from the queue,
            # go check the parents's parent up in the loop
            else:
                amr_string.append(")")
                number_of_children_to_process.pop()
                try:
                    number_of_children_to_process[-1] -= 1
                except IndexError:
                    # we have hit the root => done
                    break
        return

    def process_leafnode(self, amr, curr_node, amr_string, visited_variables, number_of_children_to_process):

        if amr.node[curr_node]["var"] is None:
            # Constant => no self-bracket
            node_string = amr.node[curr_node]["concept"]
            amr_string.append(node_string)

        else:
            # has a variable => self-bracket
            var = amr.node[curr_node]["var"]
            concept = amr.node[curr_node]["concept"]

            # need to avoid variable collision
            if var in visited_variables:
                node_string = "%s" % (var)  # e.g. :ARG0 i
            else:
                node_string = "(%s)" % (" / ".join([var, concept]))  # e.g. :ARG0 (i / i)
                visited_variables.append(var)
            amr_string.append(node_string)

        # finished with this child (leaf)
        number_of_children_to_process[-1] -= 1
        self.ask_prev_parents(number_of_children_to_process, amr_string)

        return

    def pkl_graphs(self, graphs, fname):
        with open(fname, "wb") as ao:
            logger.info("Pickling graphs into --> %s" % fname)
            pickle.dump(graphs, ao, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def gen_edge_string(g, e):
        return "%s -- %s --> %s" % (g.node[e[0]]["concept"], e[2]["label"], g.node[e[1]]["concept"])