#!/usr/bin/env python2
'''
* extract
# abbrev: $W (\w+) $N (\d+) $D(date) $IP $S(.*)
cat ... | mline=1 sep=',' find.py find <pat>
cat ... | find.py find <pat1> <pat2> ... # tree match

* format
echo -e '127.0.0.1\t80' | header='ip port' grep.py format  ssh '$ip $port'

* stat
cat ... | stat_interval=1 count_filter=0 avg_filter=0 ./find.py stat <pat> # special pattern $N
cat ... | ./find.py top <pat> # pat should include '()'
'''

import sys
import re
import os
import time
import subprocess
import string
from itertools import groupby

def cfgi(key, default):
    return int(os.getenv(key, default))

def tolist(x):
    if type(x) == list or type(x) == tuple:
        return x
    else:
        return [x]

def sh(cmd):
    Popen(cmd, shell=True)

def read(path):
    return file(path).read()

def help():
    print __doc__

def parse_ts(str):
    return time.mktime(time.strptime(str, '%Y-%m-%d %H:%M:%S'))

def safe_div(x, y):
    if y == 0:
        return -1.0
    else:
        return x/y

def find_tree(*pats):
    def build_list(matchs):
        r = []
        for x in matchs:
            if type(x) == list or type(x) == tuple:
                r.extend(x)
	    else:
                r.append(x)
        return r
    matchs = [''] * len(pats)
    for line in sys.stdin:
        for idx, pat in enumerate(pats):
            m = re.search(pat, line)
            if not m: continue
            matchs[idx] = m.groups()
            if idx == len(pats) - 1:
               result = build_list(matchs)
               if all(result):
	           print '\t'.join(result)

def find(*pats):
    sep = os.getenv('sep', '\t')
    pats = map(build_regexp, pats)
    if len(pats) > 1:
        find_tree(*pats)
        return
    pat = pats[0]
    if cfgi('mline', '0'):
        for i in re.findall(pat, sys.stdin.read(), re.M|re.S):
            print sep.join(tolist(i))
    else:
        for line in sys.stdin:
            m = re.search(pat, line)
            if m:
                print sep.join(m.groups())

import mmap
def fsearch(pat, path):
    return re.search(pat, mmap.mmap(open(path).fileno(), 0))

def format(*args):
    tpl = re.sub('\$([0-9])', r'$k\1', ' '.join(args))
    tpl = string.Template(re.sub('\$([0-9])', r'$k\1', ' '.join(args)))
    for line in sys.stdin.readlines():
        values = line.strip().split('\t')
        keys = (os.getenv('header') or '').split() or ['k%d'%(i + 1) for i in range(len(values))]
        print tpl.safe_substitute(all=line.strip(), **dict(zip(keys, values)))

def build_regexp(pat):
    special_pat = dict(N='(\d+)', D='(dddd-dd-dd dd:dd:dd.d+)'.replace('d', '\d'), IP='([0-9]+[.][0-9]+[.][0-9]+[.][0-9]+)', CV='{"[A-Z]+":(.*)}', W='(\w+)', S='.*')
    def get_special_pat(x): return special_pat.get(x, '$' + x)
    return re.sub('\$([A-Z]+)', lambda m: get_special_pat(m.group(1)), pat)
def stat(pat=''):
    pat = build_regexp(pat)
    stat_interval, count_filter, avg_filter = cfgi('stat_interval', '1'), cfgi('count_filter', '0'), cfgi('avg_filter', '0')
    last_report, last_count, last_accu = 0.0, 0, 0.0
    cur_count, cur_accu = 0, 0.0
    for match in re.findall(r'(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d).*?%s'%(pat), sys.stdin.read()):
        if type(match) == tuple:
            ts, val = match[0], match[1]
        else:
            ts, val = match, 0
        cur_ts = parse_ts(ts)
        cur_accu += int(val)
        cur_count += 1
        if cur_ts - last_report > stat_interval - 0.1:
            count = cur_count - last_count
            avg = safe_div(cur_accu - last_accu, count)
            if count > count_filter and avg >= avg_filter:
                print '%s\t%d\t%f'%(ts, count, avg)
                last_report, last_count, last_accu = cur_ts, cur_count, cur_accu

def top(pat):
    pat = build_regexp(pat)
    time_key_list = re.findall(r'2017-(\d\d-\d\d \d\d:\d\d:\d\d).*%s'%(pat), sys.stdin.read())
    for time, key in sorted(time_key_list, key = lambda x: int(x[1]), reverse=True):
        print time, key

if __name__ == '__main__':
    len(sys.argv) >= 2  or help() or sys.exit(1)
    func = globals().get(sys.argv[1])
    callable(func) or help() or sys.exit(2)
    ret = func(*sys.argv[2:])
    if ret != None:
        print ret
