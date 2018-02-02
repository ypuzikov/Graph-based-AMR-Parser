#!/bin/bash

BASEDIR=$(dirname $0)
cd $BASEDIR/config_files

source env_config.sh

INPUT=$(greadlink -f $1)
INPUT_DIR=$(dirname $INPUT)

# 1.Run Stanford CoreNLP tool to get NE, UPOS, Universal dependencies
#edu/stanford/nlp/models/parser/nndep/english_UD.gz
cd $CORENLP_HOME
# java -mx6000m -cp "stanford-corenlp-3.6.0.jar:stanford-corenlp-3.6.0-models.jar:*" \
# edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,depparse \
# -depparse.model edu/stanford/nlp/models/parser/nndep/english_SD.gz -depparse.extradependencies MAXIMAL \
# -tokenize.options "normalizeParentheses=false" \
# -ssplit.eolonly \
# -file $INPUT \
# -outputFormat conll -outputDirectory $INPUT_DIR -outputExtension .sf.conll

java -mx6000m -cp "stanford-corenlp-3.6.0.jar:stanford-corenlp-3.6.0-models.jar:*" \
edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,depparse \
-tokenize.options "normalizeParentheses=false" \
-ssplit.eolonly \
-file $INPUT \
-outputFormat conll \
-outputDirectory $INPUT_DIR \
-outputExtension .conll.sf

# 2. Run Bjorkelund SRL system to get lemma, POS, dep and semantic roles
# output is conll file of the format:
# id-1 form-2 lemma-3 .-4 pos-5 .-6 .-7 .-8 head-9 .-10 dep-11 .-12 Y(predicate)-13 sense-14 p1ARG-15 p2ARG-16 p3ARG-17 ....

cd $SRL_HOME/scripts
./parse_full.sh $INPUT.conll.sf $INPUT.conll.mate

# Combining all annotation layers into one file:
#paste -d "\t" <(cut -f 1-3,6,9,11 $INPUT.conll.mate) <(cut -f 5-7 $INPUT.conll.sf) <(cut -f 13- $INPUT.conll.mate) > $INPUT.conll
paste -d "\t" <(cut -f 1- $INPUT.conll.sf) <(cut -f 3,5,9,11,13- $INPUT.conll.mate) > $INPUT.conll

echo "Preprocessing finished!"
# output format - tsv CONLL: # ID-1 FORM-2 LEMMA-3 POS-4 HEAD_MALT-5 DEP_MALT-6 NE-7 HEAD_UD-8 DEP_UD-9 Y(predicate)-10 sense-11 ARG-12 ARG-13 ARG-14 ....
