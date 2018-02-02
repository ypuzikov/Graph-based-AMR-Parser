#!/usr/bin/env python
# encoding=utf-8

"""
The script is courtesy to Didzis Gosko, Guntis Bardzins:
https://github.com/didzis/CAMR/blob/wrapper/preprocess.py

"""

# for Python 3 compatibility
from __future__ import print_function

import sys, re, os, json

# if sys.version_info.major < 3 or sys.version_info.minor < 3:
#     print("Error: requires Python 3.3+", file=sys.stderr)
#     sys.exit(1)

if sys.version_info.major < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')


try:
    money = []
    money.extend('dollars|dollar|penny|pence|cents|cent|pounds|pound|euros|euro|francs|franc|centimes|centime|kronas|krona|rubles|ruble'.split('|'))
    money.extend('pesoes|peso|centavoes|centavo'.split('|'))
    symbols = list('£$€')
    symbol2money = {
            '£': ['pound', 'pounds'],
            '€': ['euro', 'euros'],
            '\$': ['dollar', 'dollars'],
    }

    amounts = []
    amounts.extend('millions|million|billion|billions|trillion|trillions|thousand|thousands|hundred|hundreds'.split('|'))
    amount2zeros = {
            'million': '000000',
            'millions': '000000',
            'billion': '000000000',
            'billions': '000000000',
            'trillion': '000000000000',
            'trillions': '000000000000',
            'thousand': '000',
            'thousands': '000',
            'hundred': '00',
            'hundreds': '00',
    }

    word2num = {
            'zero': '0',
            # 'oh': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',

            'ten': '10',
            'eleven': '11',
            'twelve': '12',
            'thirteen': '13',
            'fourteen': '14',
            'fifteen': '15',
            'sixteen': '16',
            'seventeen': '17',
            'eighteen': '18',
            'nineteen': '19',

            'twenty': '20',
            'thirty': '30',
            'forty': '40',
            'fifty': '50',
            'sixty': '60',
            'seventy': '70',
            'eighty': '80',
            'ninety': '90',

            # 'hundred': '100',
            # 'thousand': '1000',
            # 'million': '1000000',
            # 'billion': '1000000000',
            # 'trillion': '1000000000000',

            # 'point': '.',
    }

    numord2num = {
            '0th': '0',
            '1st': '1',
            '2nd': '2',
            '3rd': '3',
            '4th': '4',
            '5th': '5',
            '6th': '6',
            '7th': '7',
            '8th': '8',
            '9th': '9',
            '11th': '11',
            '12th': '12',
            '13th': '13',
    }

    wordord2num = {

            'zeroth': '0',
            'first': '1',
            'second': '2',
            'third': '3',
            'fourth': '4',
            'fifth': '5',
            'sixth': '6',
            'seventh': '7',
            'eighth': '8',
            'ninth': '9',

            'tenth': '10',
            'eleventh': '11',
            'twelfth': '12',
            'thirteenth': '13',
            'fourteenth': '14',
            'fifteenth': '15',
            'sixteenth': '16',
            'seventeenth': '17',
            'eighteenth': '18',
            'nineteenth': '19',

            'twentieth': '20',
            'thirtieth': '30',
            'fortieth': '40',
            'fiftieth': '50',
            'sixtieth': '60',
            'seventieth': '70',
            'eightieth': '80',
            'ninetieth': '90',

            'hundredth': '100',
            'thousandth': '1000',
            'millionth': '1000000',
            'billionth': '1000000000',
            'trillionth': '1000000000000',
    }

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

            'jan': '01', # added later (Puzikov Yevgeniy 2016.07.04)
            'feb': '02',
            'mar': '03',
            'apr': '04',
            'jun': '06',
            'jul': '07',
            'aug': '08',
            'sep': '09',
            'oct': '10',
            'nov': '11',
            'dec': '12',
    }


    def times_1000_ws(m):
        return m.group(1)[0]+str(float(m.group(1)[1:])*1000).rstrip('0').rstrip('.')
        # return m.group(1)[0]+str(float(m.group(1)[1:])*1000).rstrip('0').rstrip('.')+('.' if m.group().endswith('.') else '')

    def times_1000(m):
        return str(float(m.group(1))*1000).rstrip('0').rstrip('.')

    def times_amounts(m):
        return str(float(m.group(1).strip())*float('1'+''.join(amount2zeros[x.lower()] for x in m.group(2).split(' ') if x))).rstrip('0').rstrip('.')

    def normalize_number(m):
        prefix = m.group(1)
        number = m.group(2).replace(' ', '').replace(',', '')
        if m.group(0).endswith('k'):
            number = str(float(number)*1000).rstrip('0').rstrip('.')
        if m.group(0).endswith('m'):
            number = str(float(number)*1000000).rstrip('0').rstrip('.')
        return prefix+number
        # return str(float(m.group(1).strip())*float('1'+''.join(amount2zeros[x.lower()] for x in m.group(2).split(' ') if x))).rstrip('0').rstrip('.')

    amountsre = re.compile(r'('+r'|'.join(amounts)+r')', re.I)

    def multiply_number(m):
        prefix = m.group(1)
        number = float(m.group(2))
        multiply = float('1'+''.join(amount2zeros[x.lower()] for x in amountsre.findall(m.group(3))))
        number = str(number*multiply).rstrip('0').rstrip('.')
        return prefix+number
        # float('1'+''.join(amount2zeros[x.lower()] for x in m.group(2).split(' ') if x))
        # return str(float(m.group(1).strip())*float('1'+''.join(amount2zeros[x.lower()] for x in m.group(2).split(' ') if x))).rstrip('0').rstrip('.')
        # return ''

    def multiply_two_numbers(m):
        prefix = m.group(1)
        first_number = float(m.group(2))
        conj = m.group(3)
        second_number = float(m.group(4))
        multiply = float('1'+''.join(amount2zeros[x.lower()] for x in amountsre.findall(m.group(5))))
        first_number = str(first_number*multiply).rstrip('0').rstrip('.')
        second_number = str(second_number*multiply).rstrip('0').rstrip('.')
        return prefix+first_number+conj+second_number

    def numordinal_to_number(m):
        prefix_number = m.group(1)
        ordinal = numord2num[m.group(2).lower()]
        return prefix_number+ordinal

    def normalize_date(m):
        year = m.group('year') or '0000'
        month = month2num[m.group('month').lower()]
        day = int(m.group('day') or 0)
        # return '%s%s%02i' % (year[2:], month, day)
        return '%s-%s-%02i' % (year, month, day)

    def normalize_date_interval(m):
        year_from = m.group('year_from') or '0000'
        month_from = month2num[m.group('month_from').lower()]
        day_from = int(m.group('day_from') or 0)
        year_to = m.group('year_to') or '0000'
        month_to = month2num[(m.group('month_to') or m.group('month_from')).lower()]
        day_to = int(m.group('day_to') or 0)
        middle = m.group('middle')
        # return '%s%s%02i' % (year[2:], month, day)
        return '%s-%s-%02i%s%s-%s-%02i' % (year_from, month_from, day_from, middle, year_to, month_to, day_to)

    def normalize_numeral_date(m):
        # year = int(m.group('year') or 0)
        year = m.group('year') or '0000'
        year = int(year) if len(year) == 4 else 2000+int(year)
        if year > 2020:
            year -= 100
        month = int(m.group('month') or 0)
        day = int(m.group('day') or 0)
        # return '%02i%02i%02i' % (year, month, day)
        return '%04i-%02i-%02i' % (year, month, day)

    def normalize_year(m):
        year = m.group('year')
        # return '%s0000' % (year[2:])
        return '%s-00-00' % (year)

    # def normalize_month(m):
    #     year = m.group('year')
    #     return '%s0000' % (year[2:])

    numre = re.compile(r'(?:^|(?<=\W))('+'|'.join(set(word2num.keys())|set(wordord2num.keys()))+')(?=\W|$)', re.I)

    def words_to_number(m):
        numbers = []
        last = ''
        for num in numre.findall(m.group(0)):
            num = num.lower()
            if num in wordord2num:
                num = wordord2num[num]
                if num[-1] == '1':
                    ext = 'st'
                elif num[-1] == '2':
                    ext = 'nd'
                elif num[-1] == '3':
                    ext = 'rd'
                else:
                    ext = 'th'
                if len(num) == 1 and len(last) == 2 and last[-1] == '0':
                    numbers[-1] = last[0:1]+num+ext
                else:
                    numbers.append(num+ext)
                last = num
            else:
                num = word2num[num]
                if len(num) == 1 and len(last) == 2 and last[-1] == '0':
                    numbers[-1] = last[0:1]+num
                else:
                    numbers.append(num)
                last = num
        return ''.join(numbers)

    from monthstats import extract_features, match, load_rules

    month_rules = load_rules()

    def normalize_month(m):
        global month_rules
        if not month_rules:
            return m.group(0)
        before = m.group(1)
        after = m.group(3)
        month = month2num[m.group(2).lower()]
        features = extract_features(before, after, month)
        if features:
            cls = match(month_rules, features)
        else:
            cls = 'True'
        if cls == 'True':
            return m.group(1)+'0000-'+month+'-00'+m.group(3)
        return m.group(1)+m.group(2)+m.group(3)

    fixres = [
        (
            "normalize bullet at beginning of sentence",
            re.compile(r'^(·)\s*(?=\W)', re.I),
            ''
        ),
        (
            "normalize bullets in middle",
            re.compile(r'(?<=[a-z])\s*·\s*(?=[a-z])', re.I),
            ' '
        ),
        (
            "normalize bullets in middle",
            re.compile(r'·', re.I),
            ';'
        ),
        (
            "normalize dots",
            re.compile(r'([…])', re.I),
            '...'
        ),
        (
            "normalize dashes",
            re.compile(r'([–—])', re.I),
            '-'
        ),
        # normalize quotes
        (
            "normalize quotes",
            re.compile(r'([‘’`\']{2}|[“”])', re.I),
            '"'
        ),
        (
            "normalize single quote",
            re.compile(r'([‘’`])', re.I),
            "'"
        ),
        (
            "words to numerals (incl. ordinal - they will end with st|nd|rd|th)",
            re.compile(r'(?:(?<=\W)|^)' + \
                    r'('+'|'.join(set(word2num.keys())|set(wordord2num.keys()))+r')' + \
                    r'([\s-]?(?:'+'|'.join(set(word2num.keys())|set(wordord2num.keys()))+r'))*' + \
                r'(?!\w|\-|'+'|'.join(set(word2num.keys())|set(wordord2num.keys()))+')', re.I),
                # r'(?=\W|$)', re.I),
            words_to_number
        ),
        (
            "normalize numeral 6 digit date from-to",
            re.compile(r'(?:(?<=\W)|^)(\d{6})-(\d{6})(?=\W|$)', re.I),
            r'\1 - \2'
        ),
        (
            "normalize numeral 8 digit date from-to",
            re.compile(r'(?:(?<=\W)|^)(\d{8})-(\d{8})(?=\W|$)', re.I),
            r'\1 - \2'
        ),
        # (
        #     "normalize date from year-month-day to 6 digit numeral format",
        #     re.compile(r'(?:(?<=\W)|^)(?:19|20)?(?P<year>\d\d)-(?P<month>\d\d)-(?P<day>\d\d)(?=\W|$)', re.I),
        #     normalize_numeral_date
        # ),
        (
            "normalize date from 6 digit numeral to year-month-day format",
            re.compile(r'(?:(?<=\W)|^)(?P<year>[0198]\d)(?P<month>\d\d)(?P<day>\d\d)(?=\W|$)', re.I),
            # re.compile(r'(?:(?<=\W)|^)(?P<year>\d\d)(?P<month>\d\d)(?P<day>\d\d)(?=\W|$)', re.I),
            normalize_numeral_date
        ),
        (
            "normalize date from 8 digit numeral to year-month-day format",
            re.compile(r'(?:(?<=\W)|^)(?P<year>\d\d\d\d)(?P<month>\d\d)(?P<day>\d\d)(?=\W|$)', re.I),
            normalize_numeral_date
        ),
        # (
        #     "normalize numeral date from 8 digits to 6 digits",
        #     re.compile(r'(?:(?<=\W)|^)(\d\d\d{6})(?=\W|$)', re.I),
        #     # re.compile(r'(?:(?<=\W)|^)((?:19|20)\d{6})(?=\W|$)', re.I),
        #     lambda m: m.group(1)[2:]
        # ),
        # (
        #     "normalize numeral 6 digit date from-to",
        #     re.compile(r'(?:(?<=\W)|^)(\d{6})-(\d{6})(?=\W|$)', re.I),
        #     r'\1 - \2'
        # ),
        (
            "normalize date from day month year to year-month-day format",
            re.compile(r'(?:(?<=\W)|^)' + \
                    r'(?P<day>\d+)(?:st|nd|rd|th)?(:?\sof)?\s(?P<month>'+'|'.join(month2num.keys())+r')' + \
                    r'\s?,?\s?(?P<year>(?:\d\d\d\d)?)(?=\W|$)', re.I),
                    # r'\s?,?\s?(?P<year>(?:19|20)\d\d)(?=\W|$)', re.I),
            normalize_date
        ),
        (
            "normalize date from month [day] year to year-month-day format",
            re.compile(r'(?:(?<=\W)|^)' + \
                r'(?P<month>'+'|'.join(month2num.keys())+r')\s?(?:,)?\s?(?P<day>\d*)(?:st|nd|rd|th)?' + \
                r'\s?(?:,|of)?\s?(?P<year>(?:\d\d\d\d))(?=\W|$)', re.I),
                # r'\s?(?:,|of)?\s?(?P<year>(?:19|20)\d\d)(?=\W|$)', re.I),
            normalize_date
        ),
        (
            "normalize date interval from 'month day to month day' to 0000-month-day to 0000-month-day format",
            re.compile(r'(?:(?<=\W)|^)' + \
                r'(?P<month_from>'+'|'.join(month2num.keys())+r')\s?(?:,)?\s?(?P<day_from>\d+)(?:st|nd|rd|th)?(?P<year_from>)' + \
                r'(?P<middle> to | - )' + \
                r'(?P<month_to>(?:'+'|'.join(month2num.keys())+r')?)\s?(?:,)?\s?(?P<day_to>\d+)(?:st|nd|rd|th)?(?P<year_to>)' + \
                r'(?=\W|$)', re.I),
            normalize_date_interval
        ),
        (
            "normalize date from month day to 0000-month-day format",
            re.compile(r'(?:(?<=\W)|^)' + \
                r'(?P<month>'+'|'.join(month2num.keys())+r')\s?(?:,)?\s?(?P<day>\d+)(?:st|nd|rd|th)?(?P<year>)(?=\W|$)', re.I),
            normalize_date
        ),
        (
            "normalize date from month to 0000-month-00 format",
            re.compile(r'((?:^|(?:[\W]+\w+){0,10})\W+)(?P<month>'+'|'.join(month2num.keys())+r')(\W+(?:\w+[\W]+){0,10}|$)', re.I),
            normalize_month,
            True    # iterate
        ),
        # (
        #     "normalize year to 6 digit numeral format",
        #     re.compile(r'(?:(?<=\W)|^)(?P<year>(?:19|20)\d\d)(?=\W|$)', re.I),
        #     normalize_year
        # ),
        # (
        #     "normalize month to 6 digit numeral format",
        #     re.compile(r'(?:(?<=\W)|^)(?P<month>'+'|'.join(month2num.keys())+r')(?P<day>)(?P<year>)(?=\W|$)', re.I),
        #     normalize_date
        # ),
        # (
        #     "ordinal number to number: 1st to 1, 2nd to 2, ...",
        #     re.compile(r'((?:^|\W)(?:\d*))('+'|'.join(numord2num.keys())+')(?=\W|$)', re.I),
        #     numordinal_to_number
        # ),
        (
            "insert space between a number and million, billion, pounds etc. and any other word consisting of at least two symbols",
            re.compile(r'((?:^|\W)\d+)('+'|'.join(money+amounts)+'|[^\W\d]{3,})', re.I),
            r'\1 \2'
        ),
        (
            "number normalization, can start with currency symbol, if ends with k or m then multiply by 1000 or 1000000",
            re.compile(r'(?:(?<=[^\w\-])|^)((?:(?:'+'|'.join(symbols)+')\s?)?)' + \
                    r'((?:\d[,\s]\d{2,3}|\d)+(?:\.\d*\d)?)' + \
                    r'[km]?(?=\W|$)', re.I),
            normalize_number
        ),
        (
            "from <number> million billions to <number>00000...",
            re.compile(r'((?<=\W)(?:between|from|till|with)\s+|^)(\d+\.\d+|\d+)(\s*(?:and|to)\s*)(\d+\.\d+|\d+)((?:\s?(?:'+'|'.join(amounts)+'))+)(?=\W|$)',
                re.I),
            multiply_two_numbers
        ),
        (
            "from <number> million billions to <number>00000...",
            re.compile(r'(?:(?<=\W)|^)((?:'+'|'.join(symbols)+')?)(\d+\.\d+|\d+)((?:\s?(?:'+'|'.join(amounts)+'))+)(?=\W|$)', re.I),
            multiply_number
        ),
    ]

    for symbol, m in symbol2money.items():
        # $1 => 1 dollar
        fixres.append((
            "symbol2money: "+symbol+" : $1 => 1 dollar",
            re.compile(symbol+r'1(\.0+)?(?=\W|$)', re.I),
            r'1 '+m[0]
        ))
        # $100 => 100 dollars
        fixres.append((
            "symbol2money: "+symbol+" : $100 => 100 dollars",
            re.compile(symbol+r'\s*(\d+\.?\d+|\d+)(?=\W|$)', re.I),
            r'\1 '+m[1]
        ))
        # $s => dollars
        fixres.append((
            "symbol2money: "+symbol+" : $s => dollars",
            re.compile(symbol+r's(?=\W|$)', re.I),
            m[1]
        ))


    def preprocess_line(line):
        i = 0
        # for title,r,s,*extra in fixres:
        for entry in fixres:
            title = entry[0]
            r = entry[1]
            s = entry[2]
            extra = entry[3:]
            before = line
            if extra and extra[0]:
                start_line = line
                while True:
                    line = r.sub(s, line, 0)
                    if line == start_line:
                        break
                    start_line = line
            else:
                line = r.sub(s, line, 0)
            if before != line:
                repl_statistics[i][1].append([before, line])
            i += 1

        return line


    repl_statistics = []
    # for title,r,s,*extra in fixres:
    for entry in fixres:
        title = entry[0]
        repl_statistics.append([title, []])

    def output_statistics(f=sys.stderr):
        for i,stats in enumerate(repl_statistics):
            print('#%i ========== %s ==========' % (i, stats[0]), file=f)
            print(file=f)
            for repl in stats[1]:
                print(("<<< %s" % repl[0]), file=f)
                print((">>> %s" % repl[1]), file=f)
                print(file=f)
            print(file=f)
            print(file=f)


    if __name__ == "__main__":

        def usage():
            print('usage:', sys.argv[0], '[--plain] [--apply-on <comment tag>] [input.amrs... | < input.amr] > output.amr', file=sys.stderr)

        args = sys.argv[1:]

        if '--help' in args or '-h' in args:
            usage()
            sys.exit(0)

        apply_on = 'snt'
        plain = False
        debug = False

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == '--plain' or arg == '-p':
                args.pop(i)
                plain = True
            elif arg == '--debug' or arg == '-d':
                args.pop(i)
                debug = True
            elif arg == '--apply-on' or arg == '-a':
                args.pop(i)
                apply_on = args.pop(i)
            else:
                i += 1

        if not plain:
            print('Will apply on', apply_on, 'comment tags', file=sys.stderr)

        anytag = re.compile(r'[#\s]::\w+\s')
        applytag = re.compile(r'[#\s]::'+apply_on+'\s', re.U)

        def sources():
            if args:
                for arg in args:
                    with open(arg, 'rb') as f:
                        yield (line.decode('utf8', errors='replace') for line in f)
            else:
                usage()
                print('waiting for input from stdin', file=sys.stderr)
                yield sys.stdin.buffer

        wikire = re.compile(r':wiki (?:-|"[^"]*")\s*')

        line = ''
        for f in sources():
            if line:
                print() # add extra empty line between sources
            for line in f:
                if debug:
                    line_before = line
                if plain:
                    line = preprocess_line(line[:-1]) + line[-1]
                elif line.startswith('#'):
                    m = applytag.search(line)
                    if m:
                        pos = m.end()
                        m = applytag.search(line, pos)
                        if m:
                            endpos = m.start()
                        else:
                            endpos = len(line)-1    # exclude newline
                        print('# ::src-snt', line[pos:endpos].strip())
                        line = line[0:pos] + preprocess_line(line[pos:endpos]) + line[endpos:]
                else:
                    line = wikire.sub(r'', line)    # remove :wiki's
                if debug:
                    if line != line_before:
                        print("<<", line_before.strip())
                        print(">>", line.strip())
                        print()
                else:
                    print(line.rstrip())

        if debug:
            output_statistics()

except KeyboardInterrupt:
    print('Interrupted', file=sys.stderr)
    sys.exit(1)