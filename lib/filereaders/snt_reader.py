#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from base_reader import FileReader

logger = logging.getLogger("main")

class SntFileReader(FileReader):

    def read_file(self):

        logger.info("Reading input sentence file")

        data = []

        with open(self.fname, "r") as snt_file:
            for line_idx, line in enumerate(snt_file):
                datum = {"id": str(line_idx), "snt": line.strip()}
                data.append(datum)

        return data