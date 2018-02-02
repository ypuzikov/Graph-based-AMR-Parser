#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO we rely on some libs, need to explicitly mention all of them
# as of now the most important ones are those of Flanigan, Gosko and Schneider

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import lib.prp as prp
import lib.pipeline as pipelib
from lib.util import init_log
from config_files.main_config import gen_config

def parse_args():
    import argparse
    usage = "preprocess: python main.py --mode prp" \
            "train (concept identification): python main.py --mode train_ci" \
            "train (relation identification): python main.py --mode train_ri \ " \
            "parse: python main.py --mode parse "

    description = "AMR parser v1.0"
    arg_parser = argparse.ArgumentParser(description=description, usage=usage)

    arg_parser.add_argument("--mode", choices=["prp", "train_ci", "train_ri", "parse"], required=True)

    args = arg_parser.parse_args()
    mode = args.mode

    config = gen_config(mode)
    return config


def main(params):
    logger = init_log(os.path.join(params["model_dir"], "log.txt"))

    mode = params["mode"]

    if mode == "prp":
        prp.preprocess_input(params)

    elif mode == "train_ci":
        stage_parser = pipelib.CI_Stage(params, "CI")
        stage_parser.train()

    elif mode == "train_ri":
        stage_parser = pipelib.RI_Stage(params, "RI")
        stage_parser.train()

    elif mode == "parse":

        pipelib.parse_all_stages(params)

    else:
        sys.stderr.write("Check command line options!")


if __name__ == "__main__":
    parameters = parse_args()
    main(parameters)
