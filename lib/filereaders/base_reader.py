#!/usr/bin/env python
# -*- coding: utf-8 -*-

class FileReader(object):

    def __init__(self, fname, vocab, data, params=None):
        self.fname = fname
        self.data = data
        self.vocab = vocab
        self.params = params

    def read_file(self):
        pass