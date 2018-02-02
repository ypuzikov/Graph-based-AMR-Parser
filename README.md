# Graph-Based AMR Parser

An implementation of a graph-based AMR parsing system. Inspired by:

   - "A Discriminative Graph-Based Parser for the Abstract Meaning Representation" (J. Flanigan, S. Thomson, J. Carbonell, C. Dyer, N. Smith). Proc. ACL, 2014. [pdf][jamr-paper]
   - "Robust Subgraph Generation Improves Abstract Meaning Representation Parsing" (K. Werling, G. Angeli, and C. Manning), Proc. ACL, 2015. [pdf][werling-paper]
 
## Version
0.1.0

## Dependencies
The system is written in Python 2.7, relying on the following libraries:

* [numpy][np]
* [scipy][scp]
* [sklearn][sklrn]
* [matplotlib][matplot]
* [networkx][nx]
* [pattern.en][pattern]
* [parsimonious][parsimonious-lib]
* [pyjarowinkler][jaro]

The preprocessing pipeline makes use of the following software:

* [Stanford Corenlp 3.6.0][corenlp]
* [MATE Tools][mate-tools] (semantic role labelling system with pretrained model files: [srl-4.31.tgz][srl])
* [JAMR][jamr]

You will need to download them and specify the corresponding paths in the ```./scripts/config_files/env_config.sh``` file

## How to run
### Step 1
Set the environment variables in

```
/scripts/config_files/env_config.sh
```

and

```
./scripts/config_files/main_config.py
```

The first one contains paths to Stanford CoreNLP, SRL system and JAMR. The second one stores variables that are used by python scripts. You need to set the variables **before** running the system!

**Note:** 

   - ```/windroot/puzikov/AMR/GAMR/``` is the home folder for this repository. It contains both pretrained models and preprocessed datasets.
   - ```/windroot/puzikov/AMR/data/``` contains all the publicly available AMR corpora. 
   

### Step 2
Choose a mode: 

* *prp* -- preprocessing
* *train_ci* -- train concept identification stage
* *train_ri* -- train relation identification stage
* *parse* -- parse a data file (a `.pkl` file with **preprocessed** data)

To run the system, navigate to the script folder 
```
$ cd ./scripts
```
and run
```
$ python main.py --mode X
```
`X` stands for the chosen mode from the list above. If you get errors saying that a file or folder is not found -- check the variables in the config files (Step1).

### Step 3
In order to evaluate parsing results, navigate to the folder with the scoring script:
```
cd ./lib/eval/smatch_2.0.2
```
and run the following command:
```
python smatch.py -f testfile testfile.amr --pr
```
This computes precision, recall and F1 score according to the Smatch metric. **testfile.amr** is one of the files, which the parser produces when parsing **testfile**. You will find it in the respective folder under **./experiments/**

The performance of the concept identification stage alone can be computed by running the following command:

```
python ./lib/eval/eval_ci.py testfile.txt testfile.parsed --fmt isi
```
This will compute precision, recall and F1 score, measuring the accuracy of the concept identification. 

**Note:** the Parsimonious library, which is used for extracting concepts from PENMAN strings, often breaks when an input AMR is noisy or malformed. In this case you'd better clean the data before preprocessing and parsing.


## Experimental results

End-to-end performance of the model, trained on the LDC2014T12 dataset:

|Dataset           |Precision|Recall|F1 | 
| --- | --- | --- | --- |
|LDC2014T12        | 0.51 | 0.54 | 0.53 |
|BioMed            | 0.30 | 0.44 | 0.35 |
|The Little Prince | 0.44 | 0.53 | 0.48 |

Concept identification performance, measured on the LDC2014T12 dataset: 

   - precision: 0.76
   - recall: 0.73
   - F1 score: 0.74 

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [jamr-paper]: <http://www.cs.cmu.edu/~jmflanig/flanigan+etal.acl2014.pdf>
   [werling-paper]: <http://nlp.stanford.edu/pubs/2015werling-amr.pdf>
   [np]: <https://github.com/numpy/numpy/releases>
   [scp]: <https://github.com/scipy/scipy/releases>
   [sklrn]: <http://scikit-learn.org/stable/>
   [matplot]: <http://matplotlib.org/downloads.html>
   [nx]: <https://networkx.github.io>
   [pattern]: <http://www.clips.ua.ac.be/pages/pattern-en>
   [jaro]: <https://pypi.python.org/pypi/pyjarowinkler>
   [theano]: <http://deeplearning.net/software/theano/index.html>
   [corenlp]: <http://stanfordnlp.github.io/CoreNLP/>
   [mate-tools]: <https://code.google.com/archive/p/mate-tools/downloads>
   [srl]: <https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mate-tools/srl-4.31.tgz>
   [jamr]: <https://github.com/jflanigan/jamr>
   [parsimonious-lib]: <https://pypi.python.org/pypi/parsimonious/>
