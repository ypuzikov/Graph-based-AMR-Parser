#!/usr/bin/env python3
# encoding=utf-8

"""
The script is courtesy to Didzis Gosko, Guntis Bardzins:
https://github.com/didzis/CAMR/blob/wrapper/monthstats.py

"""

# for Python 3 compatibility
from __future__ import print_function

import sys, os, re, json
from collections import defaultdict


homedir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
monthstats = os.path.abspath(os.path.join(homedir, "monthstats.json"))

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    def __getattribute__(self, key):
        try:
            return super(Dict, self).__getattribute__(key)
        except:
            return
    def __delattr__(self, name):
        if name in self:
            del self[name]

splitre = re.compile(r'\W+', re.I)
numre = re.compile(r'^\d+$', re.I)

def extract_features(before, after, month):
    if type(before) is str:
        before = list(x for x in splitre.split(before) if x and not numre.match(x))
    else:
        before = list(x for x in before if x and not numre.match(x))
    if type(after) is str:
        after = list(x for x in splitre.split(after) if x and not numre.match(x))
    else:
        after = list(x for x in after if x and not numre.match(x))
    features = {}
    for w in before:
        features["L:"+w] = '+'
        features[":"+w] = '+'
    for w in after:
        features["R:"+w] = '+'
        features[":"+w] = '+'
    features["M:"+month] = '+'
    return features

def load_rules(filename=monthstats):
    try:
        with open(filename) as f:
            settings = json.load(f, object_hook=Dict)
            return settings.rules
    except KeyboardInterrupt:
        raise
    except:
        print('Unable to load month ruleset', file=sys.stderr)

def match(rules, features):
    results = defaultdict(float)
    for rule in rules:
        found = True
        for k,v in rule.data.items():
            if features.get(k) != v:
                found = False
                break
        if found:
            results[rule['class']] += rule.w
    max_score = 0
    max_cls = None
    for cls,score in results.items():
        if max_cls is None or score > max_score:
            max_cls = cls
            max_score = score
    return max_cls

if __name__ == "__main__":

    import pyd21 as PyC60
    import smatch_api

    args = sys.argv[1:]

    month2num = {
            'january': '01',
            'february': '02',
            'march': '03',
            'april': '04',
            'may': '05',
            'june': '06',
            'july': '07',
            'august': '08',
            'september': '09',
            'october': '10',
            'november': '11',
            'december': '12',
    }

    # monthnum = re.compile(r'((?:^|(?:[\W]+\w+){0,10})\W+)0000-(\d\d)-(\d\d)(\W+(?:\w+[\W]+){0,10}|$)', re.I)
    monthnum = re.compile(r'(?P<left>(?:^|(?:[\W]+\w+){0,10})\W+)(?P<month>'+'|'.join(month2num.keys())+r')(?P<day>)(?P<right>\W+(?:\w+[\W]+){0,10}|$)',
            re.I)

    true_before = set()
    false_before = set()
    true_after = set()
    false_after = set()
    true_counts_before = defaultdict(int)
    false_counts_before = defaultdict(int)
    true_counts_after = defaultdict(int)
    false_counts_after = defaultdict(int)

    data = []

    datafn = 'monthstats.json'
    if os.path.isfile(datafn):
        with open(datafn) as f:
            settings = json.load(f, object_hook=Dict)
            # data = json.load(f, object_hook=Dict)
            if set(settings.files) == set(args):
                data = settings.data
                args = settings.files

    if not data:
        for fn in args:
            print('Reading', fn, '...')
            with open(fn) as f:
                for amr in smatch_api.parse_amr_iter(f):
                    # print(amr.text)
                    startpos = 0
                    # for m in monthnum.finditer(amr.text):
                    while True:
                        m = monthnum.search(amr.text, startpos)
                        if not m:
                            break
                        # before = list(x for x in m.group(1).split(' ') if x)
                        # after = list(x for x in m.group(4).split(' ') if x)
                        # before = list(x for x in splitre.split(m.group(1)) if x)
                        # after = list(x for x in splitre.split(m.group(4)) if x)
                        before = list(x for x in splitre.split(m.group('left')) if x)
                        after = list(x for x in splitre.split(m.group('right')) if x)
                        # month = m.group(2)
                        month = m.group('month')
                        month = month2num[month.lower()]
                        # day = m.group(3)
                        day = m.group('day')
                        # startpos = m.end(2) # continue right after previously found month
                        startpos = m.end('month') # continue right after previously found month
                        is_month = bool(re.search(r'date-entity [()]*:month '+str(int(month)), amr.amr_line))
                        data.append(Dict(before=before, after=after, is_month=is_month, month=month, day=day))
                        print(month+'-'+day, is_month, ': ', before, '  ::::::  ', after)
                # print(amr.amr_line)
            # print(amr.amr_line)
        with open(datafn, 'w') as f:
            json.dump(dict(files=args, data=data), f, ensure_ascii=False, indent=2)

    if not data:
        print('no input files specified', file=sys.stderr)
        sys.exit(1)

    classifier = PyC60.Classifier()
    classifier.max_features = 4
    # classifier.filter_rules = True

    skipped = 0

    for item in data:
        features = extract_features(item.before, item.after, item.month)
        if features:
            classifier.add(str(item.is_month), features)
        elif item.is_month:
            skipped += 1

        # before_set = set(item.before)
        # after_set = set(item.after)
        # if item.is_month:
        #     for w in before_set:
        #         true_counts_before[w] += 1
        #     for w in after_set:
        #         true_counts_after[w] += 1
        #     true_before |= before_set
        #     true_after |= after_set
        # else:
        #     for w in before_set:
        #         false_counts_before[w] += 1
        #     for w in after_set:
        #         false_counts_after[w] += 1
        #     false_before |= before_set
        #     false_after |= after_set
        # print(item.month+'-'+item.day, item.is_month, ': ', item.before, '  ::::::  ', item.after)

    classifier.train()
    rules = classifier.rules()

    with open(datafn, 'w') as f:
        json.dump(dict(files=args, data=data, rules=rules), f, ensure_ascii=False, indent=2)

    import pprint
    # pprint.pprint(classifier.ruleset())
    # print(classifier.ruleset())
    classifier.print_classes()
    classifier.print()
    classifier.print_classes()

    print(skipped, 'true classes skipped')

    # print('True before:', ','.join(sorted(true_before)))
    # for k,c in sorted(true_counts_before.items(), key=lambda item: item[1], reverse=True):
    #     print(k, '=', c)
    # print('True after:', ','.join(sorted(true_after)))
    # for k,c in sorted(true_counts_after.items(), key=lambda item: item[1], reverse=True):
    #     print(k, '=', c)
    # print('False before:', ','.join(sorted(false_before)))
    # for k,c in sorted(false_counts_before.items(), key=lambda item: item[1], reverse=True):
    #     print(k, '=', c)
    # print('False after:', ','.join(sorted(false_after)))
    # for k,c in sorted(false_counts_after.items(), key=lambda item: item[1], reverse=True):
    #     print(k, '=', c)