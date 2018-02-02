#!/bin/bash

BASEDIR=$(dirname $0)
cd $BASEDIR/config_files
source env_config.sh

# input should be tokenized AMR file, which has :tok tag in the comments!
INPUT=$(greadlink -f $1)
OUTPUT=$2

# JAMR config
cd $JAMR_HOME/scripts
source config.sh

# Align input file
$JAMR_HOME/run Aligner -v 0 < $INPUT > $OUTPUT 2>$OUTPUT.log