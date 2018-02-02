#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import logging
import os
import operator
import re
import random
import sys
import string
import time

logger = logging.getLogger("main")

class Cleaner(object):
    def __init__(self):
        pass

    def process_digit1(self, word, snt):

        if len(word) >= 2:

            if word[-1].isalpha() or (word[-2].isalpha() and word[-1] in string.punctuation + "''"):

                # splitting ddd-wwww
                if "-" in word:
                    # trying to cure things like 4000-6000mg
                    if re.match("\d+-\d+\w+", word):
                        digit_item = "-".join([re.sub("\D", "", x) for x in word.split("-")])
                        char_item = re.sub("[^a-zA-Z]+", "", word)
                        snt.append(digit_item)
                        snt.append(char_item)

                    else:
                        # curing ddd-wwww
                        items = word.split("-")
                        for i in items:
                            snt.append(i)

                # # curing bad date patterns - don't touch it!
                # Gosko's script takes care of it
                # elif date_th_pat.match(word) is not None:
                #     modify_th(toks, word, widx)

                # skipping good date and ordinal patterns
                elif re.match("\d+('s|\ds|rd|th|nd|st).?", word) is not None:
                    snt.append(word)

                # skipping decimals
                elif "/" in word:
                    snt.append(word)

                # curing bad patterns
                else:
                    items = [x for x in re.split("(\d+)", word) if x is not None and len(x) >= 1]
                    for i in items:
                        snt.append(i)

            # these will be all sorts of multi-integer numbers
            else:
                snt.append(word)

                # these will be all sorts of one-integer numbers
        else:
            snt.append(word)

    def process_digit2(self, word, snt):

        if re.match("\(\d+-[^\d]\w+\)", word) is not None:
            items = word.split("-")
            for i in items:
                snt.append(i)
        else:
            snt.append(word)

    def process_hyph(self, word, snt,hyph_stoplist):

        # curing crazy hyphenation in the training data
        parts = word.split("-")

        # # fixme BioMed?
        # if len(parts) == 2 and parts[1] == "":
        #     snt.append(parts[0])

        # case 1: names
        if word[0].isupper():
            snt.append(word)

        else:
            # case 1: prefix in ["up", "to", "ex","no", "non"]
            two_char_parts = [x for x in parts if len(x) <= 2]
            split_prefs = [x for x in two_char_parts if x.lower() in ["up", "to", "ex", "so", "no"]]

            if len(two_char_parts) > 0:
                if len(split_prefs) > 0:
                    for i in parts:
                        snt.append(i)

                # case 2: prefix = "re".
                # This one is tricky: if the 2nd part starts with "e", it is hyphenated.
                elif "re" in two_char_parts:
                    try:
                        if parts[1][0] == "e":
                            snt.append(word)

                        else:
                            snt.append(re.sub("-", "", word))
                    except IndexError:
                        # re-
                        logger.info("Strange hyphenation pattern: %s" % word)
                        snt.append(parts[0])

                else:
                    snt.append(word)

            elif word in hyph_stoplist:
                snt.append(word)

            else:
                for i in parts:
                    snt.append(i)

    def process_slash(self, word, snt):

        if "http" in word:
            snt.append(word)

        elif re.match("\w+/\w+", word):
            # print "Splitting /:", word
            for i in word.split("/"):
                snt.append(i)

        else:
            # print "UNK /:", word
            snt.append(word)

    def clean_tokens(self, toks, hyph_stoplist):

        snt_len = len(toks)
        snt = []

        for widx, word in enumerate(toks):

            if word[0].isdigit():
                self.process_digit1(word, snt)

            # (300-mile)
            elif len(word) >= 2 and word[1].isdigit():
                self.process_digit2(word,snt)

            elif "-" in word:
                self.process_hyph(word,snt,hyph_stoplist)

            # elif word[0] == "@":

            elif "/" in word:
                self.process_slash(word,snt)

            # elif re.match("\w+\d+", word) is not None:
            #     print "word and digit", word
            #     snt.append(word)

            # elif re.match("\w+", word) is None:
            #     print word

            else:
                # print word
                snt.append(word)

        return snt

    def clean_line(self, line):

        # for the BioMed corpus
        # stripping html/xml tags
        # "<xref.*?>AAA</xref>.*" -> AAA
        l = re.sub("<.*?>", "", line.strip())
        l = re.sub("Greenwich Mean Time", "GMT", l)

        # # for normal corpora
        # # strangely enough, all mentions of "Greenwich Mean Time" should be converted to GMT
        # l = re.sub("Greenwich Mean Time", "GMT", line.strip())

        return l

    def clean_input_amr(self, in_fn, out_fn, hyph_stoplist, hyph_trunc_stoplist):

        """ A de-noising procedure to clean the AMR data """

        logger.info("Cleaning input file")
        with open(in_fn) as infile:
            out = open(out_fn, "w")
            for line in infile:

                # considering only ::snt lines
                if line.startswith("# ::snt"):
                    line = self.clean_line(line)
                    toks = line.split()[2:]
                    snt = self.clean_tokens(toks, hyph_stoplist)
                    sent = " ".join(snt)
                    out.write("# ::snt %s\n" %(sent))

                else:
                    out.write("%s" %line)

            out.close()
            logger.info("Done cleaning input file")

    def clean_input_snt(self, in_fn, out_fn, hyph_stoplist, hyph_trunc_stoplist):

        """ Same cleaning procedure, but for input in the sentence-per-line format """

        with open(in_fn) as infile:
            out = open(out_fn, "w")
            for line in infile:
                line = self.clean_line(line)

                toks = line.split()
                snt = self.clean_tokens(toks, hyph_stoplist)
                sent = " ".join(snt)
                out.write("%s\n" % (sent))

            out.close()
            logger.info("Done cleaning input file")
