#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import collections
import logging
import operator
import os
import random
import subprocess
import sys

import rules
from data_normalization import clean_data as cd
from filereaders.snt_reader import SntFileReader
from filereaders.conll_reader import ConllTrainFileReader,ConllTestFileReader
from filereaders.amr_reader import JamrFileReader
from util import rec_dd

logger = logging.getLogger("main")

ruleset = rules.Ruleset()
dcleaner = cd.Cleaner()

class FilePreprocessor(object):

    def __init__(self, vocab, params = None):

        self.vocab = vocab
        self.params = params

        self.script_dir = params["script_dir"]
        self.stats_dir = params["stat_dir"]
        self.lib_dir = params["lib_dir"]
        self.norm_script = params["norm_script"]

    def gen_amr_fnames(self, fname):

        logger.info("* AMR * format file ")

        self.fname = os.path.abspath(fname)
        self.clean_fname = "%s.clean" % self.fname
        self.norm_fname = "%s.norm" % self.clean_fname
        self.snt_fname = "%s.snt" % self.norm_fname
        self.conll_fname = "%s.conll" % self.snt_fname
        self.amr_tok_fname = "%s.amr.tok" % self.norm_fname
        self.jamr_aligned_fname = "%s.jamr-aligned" % self.amr_tok_fname
        self.isi_aligned_fname = "%s.isi-aligned" % self.amr_tok_fname
        self.vconcept_list_fn = "%s.vconcepts" % self.jamr_aligned_fname

    def gen_snt_fnames(self, fname):

        logger.info("* SENTENCE * format file")

        self.fname = os.path.abspath(fname)
        self.clean_fname = "%s.clean" % self.fname
        self.norm_fname = "%s.norm" % self.clean_fname
        self.conll_fname = "%s.conll" % self.norm_fname

    def process_amr_file(self, fname, normalize=True):
        pass

    def process_snt_file(self, fname, normalize=True):
        pass

    def align_jamr(self):

        logger.info("Aligning using JAMR")
        cmd3 = "zsh %s/jamr_align.sh %s %s" % (self.script_dir, self.amr_tok_fname, self.jamr_aligned_fname)
        subprocess.call(cmd3, shell=True)

    def gen_vconcept_list(self):

        logger.info("Extracting verb concept list")
        cmd4 = 'grep "^# ::node" %s|cut -f3|grep "^[a-z].*[0-9][0-9]$" > %s' % (self.jamr_aligned_fname, self.vconcept_list_fn)
        subprocess.call(cmd4, shell=True)

    def gen_vconcept_dict(self):

        """
        We need to create a dictionary of verb concepts (VIDX).
        VIDX[key] = [value1, value2, value3, ...]
        Here "key" is a 2-letter prefix of a verb, "value" is a verb-num1num2

        We have:
        - a Propbank list of verb concepts (PVC) with their senses (prop_vconcept_dict),

        We build:
        1. a dictionary of verb concepts, paired with the most freq sense (according to the train data) (VCF)
        2. a union of PVC and VCF -> VIDX

        VIDX is used to generate concept candidates for verbs.

        :param vconcept_list_fname: a file (one "verb-sense" pair per line), extracted from the train data
        :param prop_vconcept_dict: an attribute of the Ruleset, created when loading all databases.
        :return: VIDX

        """

        def sanity_check(pvc, vcf):
            err = False
            err_verbs = []
            for verb, sense in vcf.items():
                if verb not in pvc:
                    err_verbs.append(verb)

            if err:
                print "Some verb concepts from the training data " \
                      "were not found in PropBank: %s" % (" ".join(err_verbs))

                sys.exit()

            logger.debug("Sanity check passed (pvc, vcf)")

        VCF = {}

        # build a freq dict from training data (VCF)
        with open(self.vconcept_list_fn) as vlf:

            verb_vocab = collections.defaultdict(rec_dd)

            for line in vlf:
                items = line.strip().split("-")
                if len(items) > 2:
                    continue

                base = items[0]
                sense = items[-1]
                verb_vocab[base][sense] = 1 + verb_vocab[base].get(sense, 0)

            for verb, senses in verb_vocab.items():
                most_common_sense = max(senses.items(), key=operator.itemgetter(1))
                VCF[verb] = most_common_sense[0]

        PVC = ruleset.gen_propbank_vlist()

        # fixme: a sanity check - comment out later
        # making sure that all concepts from training data are actually from PropBank
        sanity_check(PVC, VCF)

        # update VCF with PVC
        VIDX = {}
        for verb, senses in PVC.items():
            pref = verb[:2]
            if verb not in VCF:
                sense = min(senses)
            else:
                sense = VCF[verb]

            VIDX.setdefault(pref, []).append("%s-%s" % (verb, sense))

        logger.debug("Generated augmented verblist: %d prefixes" % (len(VIDX)))
        random_key = random.choice(VIDX.keys())
        logger.debug("Random sample: %s %s " % (random_key, VIDX[random_key]))
        return VIDX

class TrainFileProcessor(FilePreprocessor):

    def process_amr_file(self, fname, normalize=True):

        self.gen_amr_fnames(fname)

        if normalize:

            if not os.path.exists(self.clean_fname):
                dcleaner.clean_input_amr(self.fname, self.clean_fname, ruleset.hyph_dict, ruleset.hyph_dict_prefix)

            if not os.path.exists(self.norm_fname):
                cmd0 = "python %s %s > %s" % (self.norm_script, self.clean_fname, self.norm_fname)
                subprocess.call(cmd0, shell=True)

        if not os.path.exists(self.conll_fname):
            logger.info("No CONLL file found => preprocessing with CoreNLP")

            # get sentences
            cmd1 = 'grep "^# ::snt" %s | cut -d" " -f3- > %s' % (self.norm_fname, self.snt_fname)
            subprocess.call(cmd1, shell=True)

            # preprocess sentences using CoreNLP
            cmd2 = "zsh %s/preprocess_cnlp.sh %s " % (self.script_dir, self.snt_fname)
            subprocess.call(cmd2, shell=True)

        # extracting data for each sentence
        logger.info("Reading CoreNLP output: %s " % self.conll_fname)
        conll_reader = ConllTrainFileReader(self.conll_fname, vocab=None, data=None)
        data, embedding_vocab = conll_reader.read_file()

        # processing alignments
        # write amr.tok and amr.tok.amr, amr.tok.eng (for ISI aligner)
        logger.debug("Creating tokenized AMR file")
        amr_reader = JamrFileReader(self.jamr_aligned_fname, vocab=None, data=data, params=self.params)
        amr_reader.gen_tok_file(fname=self.norm_fname, data=data)

        # get JAMR alignment
        if not os.path.exists(self.jamr_aligned_fname):
            self.align_jamr()

        if self.vocab is None:
            # get vconcept dictionary
            if not os.path.exists(self.vconcept_list_fn):
                self.gen_vconcept_list()

            amr_reader.vocab = collections.defaultdict()
            amr_reader.vocab["emb_voc"] = list(set(embedding_vocab))
            amr_reader.vocab["vidx"] = self.gen_vconcept_dict()

        # # fixme get ISI alignment:  MGIZA works on Linux, MacOS ??
        # if not os.path.exists(ISI_aligned_fname):
        #     logger.info("Need to do ISI alignment first!")
        #     sys.exit()

        amr_reader.read_file()
        return data, amr_reader.vocab

class TestFileProcessor(FilePreprocessor):

    def process_amr_file(self, fname, normalize=True):

        self.gen_amr_fnames(fname)
        assert self.vocab is not None

        if normalize:

            if not os.path.exists(self.clean_fname):
                dcleaner.clean_input_amr(self.fname, self.clean_fname, ruleset.hyph_dict, ruleset.hyph_dict_prefix)

            if not os.path.exists(self.norm_fname):
                cmd0 = "python %s %s > %s" % (self.norm_script, self.clean_fname, self.norm_fname)
                subprocess.call(cmd0, shell=True)

        if not os.path.exists(self.conll_fname):
            logger.info("No CONLL file found => preprocessing with CoreNLP")

            # get sentences
            cmd1 = 'grep "^# ::snt" %s | cut -d" " -f3- > %s' % (self.norm_fname, self.snt_fname)
            subprocess.call(cmd1, shell=True)

            # preprocess sentences using CoreNLP
            cmd2 = "zsh %s/preprocess_cnlp.sh %s " % (self.script_dir, self.snt_fname)
            subprocess.call(cmd2, shell=True)

        # extracting data for each sentence
        logger.debug("Reading CoreNLP output: %s " % (self.conll_fname))
        conll_reader = ConllTestFileReader(self.conll_fname, vocab=self.vocab["emb_voc"], data=None)
        data = conll_reader.read_file()

        # processing alignments
        # write amr.tok and amr.tok.amr, amr.tok.eng (for ISI aligner)
        amr_reader = JamrFileReader(self.jamr_aligned_fname, vocab=self.vocab, data=data,params=self.params)
        amr_reader.gen_tok_file(fname=self.norm_fname, data=data)

        # get JAMR alignment
        if not os.path.exists(self.jamr_aligned_fname):
            self.align_jamr()

        # # fixme get ISI alignment:  MGIZA works on Linux, MacOS ??
        # if not os.path.exists(ISI_aligned_fname):
        #     logger.info("Need to do ISI alignment first!")
        #     sys.exit()

        amr_reader.read_file()
        return data

    def process_snt_file(self, fname, normalize=True):

        self.gen_snt_fnames(fname)

        if normalize:
            if not os.path.exists(self.clean_fname):
                dcleaner.clean_input_snt(self.fname, self.clean_fname, ruleset.hyph_dict, ruleset.hyph_dict_prefix)

            if not os.path.exists(self.norm_fname):
                logger.info("Normalizing data ...")
                cmd0 = "python %s --plain %s > %s" % (self.norm_script, self.clean_fname, self.norm_fname)
                subprocess.call(cmd0, shell=True)

        if not os.path.exists(self.conll_fname):
            logger.info("No CONLL file found => preprocessing with CoreNLP")

            # preprocess sentences using CoreNLP
            cmd1 = "zsh %s/preprocess_cnlp.sh %s " % (self.script_dir, self.norm_fname)
            subprocess.call(cmd1, shell=True)

        logger.debug("Generating comments from %s " % (self.norm_fname))
        snt_reader = SntFileReader(self.norm_fname,vocab=None,data=None)
        snt_data = snt_reader.read_file()

        logger.debug("Reading CoreNLP output: %s " % (self.conll_fname))
        conll_reader = ConllTestFileReader(self.conll_fname, vocab=self.vocab["emb_voc"], data=snt_data)
        conll_data = conll_reader.read_file()

        for idx, datum in enumerate(conll_data):
            datum["comments"] = snt_data[idx]
            datum["comments"]["tok"] = " ".join([T["word"] for T in datum["tokens"]])

        return conll_data


def preprocess_input(parameters):

    dev_fn = parameters["dev_txt"] if "dev_txt" in parameters else None
    test_fn = parameters["test_txt"] if "test_txt" in parameters else None
    vocab_fn = parameters["vocab_fn"] if "vocab_fn" in parameters else None

    if vocab_fn is None:
        train_fn = parameters["train_txt"]
        processor = TrainFileProcessor(vocab=None,params=parameters)
        data, vocab = processor.process_amr_file(train_fn, normalize=True)

        logger.info("Saving %d training AMR instances" % len(data))
        pickle.dump(data, open(os.path.abspath("%s.instances.pkl" % (train_fn)), "w"))
        pickle.dump(vocab, open(os.path.abspath("%s.vocab.pkl" % (train_fn)), "w"))

    else:
        vocab = pickle.load(open(vocab_fn))

    if not parameters["amr_fmt"]:
        processor = TestFileProcessor(vocab, params=parameters)
        data = processor.process_snt_file(test_fn, normalize=True)

        logger.info("Saving %d test sentence instances " % len(data))
        pickle.dump(data, open(os.path.abspath("%s.instances.pkl" % (test_fn)), "w"))
        return

    for fname in [dev_fn, test_fn]:
        if fname is not None:
            processor = TestFileProcessor(vocab, params=parameters)
            data = processor.process_amr_file(fname=fname, normalize=True)

            logger.info("Saving %d %s AMR instances" % (len(data), fname.split("_")[0]))
            pickle.dump(data, open(os.path.abspath("%s.instances.pkl" % (fname)), "w"))

    logger.info("Done!")