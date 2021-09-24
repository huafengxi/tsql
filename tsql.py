#!/usr/bin/env python2
'''
tsql means execute sql over text file, which load data from stdin to sqlite database, then execute sql query and print the result.
tsql is useful because it provide some useful sqlite extend functions, mainly for statistic and plot.
Usage:
 ./tsql.py 'load data.tab; select * from t1'            # load tab seperated file
 ./tsql.py 'load !seq 10; select avg(c1) from t1'       # load bash command output
 ./tsql.py 'select /*py: print "pycode" */ * from t1 limit 1' # execute pycode
 db_path=':memory:' ./tsql.py 'load data.tab into t1(ts integer, val text); select ts,val from t1'
 echo 'select * from t1' | db_path='a.db' term=html ./tsql.py
 sep='\t' | ./tsql.py
# the 'data.tab' is a plain text file which may have multiple lines,
# each line have multiple columns, column seperator is regular expression '\t'
# the  data is load to sqlite database, according to 'table_schema'
# database loader support column types: integer real text.
# if table_schema not given:
#    default table_name='t1'
#    default column_name='c1, c2, c3...',
#    default type is integer or real or datetime or text, guessed by the first line
'''

import sys, os
import re
import time
import datetime
import urllib2
import sqlite3
import traceback

col_sep = os.getenv('sep', '\s+') # use tab as column seperator
http_img_root = os.getenv('http_root', '')
class QueryErr(Exception):
    def __init__(self, msg, obj=None):
        Exception(self)
        self.obj, self.msg = obj, msg

    def __str__(self):
        return "Query Exception: %s\n%s"%(self.msg, self.obj)

def load_file(path, globals, locals):
    def to_row(x):
        if type(x) == str or type(x) == unicode or not hasattr(x, '__getitem__'):
            return (x,)
        return x
    #if path.startswith('@'):
    #    return [to_row(i) for i in eval(path[1:], globals, locals)]
    if path == 'stdin':
        fd = sys.stdin
    elif path.startswith('http'):
        fd = urllib2.urlopen(path, timeout=3)
    elif path.startswith('!'):
        fd = os.popen(path[1:])
    else:
        fd = file(path)
    for line in fd:
        if not line.strip(): continue
        yield re.split(col_sep, line.strip())

def parse_cell(t, cell):
    def safe_float(x):
        if x == 'null' or x == '':
            return None
        try:
            return float(x)
        except TypeError,ValueError:
            return None
    def safe_int(x):
        if type(x) == str:
            x = x.split('.')[0]
        if x == 'null' or x == '':
            return None
        try:
            if x.startswith('0x'): return int(x, 16)
            return int(x)
        except TypeError,ValueError:
            return None
    def to_datetime(x):
        try:
            if '.' in x:
                return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
            else:
                return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    def to_unicode(text):
        return text.decode('utf8')
    value_parsers = dict(real=safe_float, text=to_unicode, integer=safe_int, bigint=safe_int, int=safe_int, datetime=to_datetime, boolean=bool)
    if not value_parsers.has_key(t):
        return None
    else:
        return value_parsers[t](cell)

load_row_count = 0
def parse_rows(types, rows):
    global load_row_count
    for row in rows:
        load_row_count += 1
        if 0 == (load_row_count % 100000):
            sys.stderr.write('load row: {}\n'.format(load_row_count))
        yield map(lambda type, cell: parse_cell(type, cell), types, row)

def render_result(result, header, term):
    def to_str(o):
        if type(o) == unicode:
            return o.encode('utf8')
        else:
            return str(o)
    def render_row(cols):
        return '<tr>%s</tr>'%(''.join('<td>%s</td>'%(to_str(cell)) for cell in cols))
    if len(result) <= 0:
        sys.stderr.write(' empty set.')
    else:
        if term == 'html':
            print '<table>'
            print render_row(header)
            for cols in result:
                print render_row(cols)
            print '</table>'
        else:
            sys.stderr.write('%s\n'%('\t'.join(header)))
            for cols in result:
                print '\t'.join(map(to_str, cols))

class TSchema:
    def __init__(self, sql):
        self.sql = sql
        self.tname, self.col_schema  = self.parse_table_schema(sql)
        self.col_names, self.col_types = [name for name, type in self.col_schema], [type for name, type in self.col_schema]
    def replace_sql(self):
        return 'insert or replace into %s(%s) values(%s)'%(self.tname, ','.join(self.col_names), ','.join(['?']*len(self.col_names)))
    def drop_sql(self):
        return 'drop table if exists %s'%(self.tname)
    def create_sql(self):
        return 'create table if not exists %s(%s);'%(self.tname, ','.join('%s %s'%(n,t) for n,t in self.col_schema))
    @staticmethod
    def guess_type(val):
        def is_int(val):
            try:
                int(val)
                return 'integer'
            except:
                pass
        def is_real(val):
            try:
                float(val)
                return 'real'
            except:
                pass
        def is_date(val):
            try:
                time.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
                return 'datetime'
            except:
                pass
        def is_date2(val):
            try:
                time.strptime(val, "%Y-%m-%d %H:%M:%S")
                return 'datetime'
            except:
                pass
        return is_int(val) or is_real(val) or is_date(val) or is_date2(val) or 'text'
    @staticmethod
    def guess_tschema(first_row):
       return 't1(%s)'%(','.join('c%d %s'%(i+1, TSchema.guess_type(val)) for i,val in enumerate(first_row)))
    @staticmethod
    def parse_table_schema(table_schema):
        ''''t1(c1 int, c2 int) -> t1, (('c1', 'int'), ('c2', 'int'))'''
        m = re.search('([0-9a-zA-Z]+) *\(([^)]+)\)$', table_schema.strip())
        if not m:
            raise QueryErr('table_schema syntax error[%s]'%(table_schema))
        else:
            table_name, col_schemas = m.groups()
        return table_name, [col.strip().split(' ', 2)[:2] for col in col_schemas.split(',')]

class TConn:
    def __init__(self, path, env):
        sqlite3.enable_callback_tracebacks(True)
        self.conn = extend(sqlite3.connect(path))
        self.env = env
    def close(self): self.conn.close()
    def before_execute(self, sql):
        term = os.getenv('term', 'text')
        if term == 'text':
            sys.stderr.write('# %s\n'%(sql))
    def parse_load_sql(self, sql):
        m = re.search('load (.+) into (.+)', sql)
        if m:
            return m.groups()
        m = re.search('load (.+)', sql)
        if m:
            return m.group(1), ''
        return '', ''
    def parse_pystmt(self, sql):
        m = re.search('/[*]py: (.+?)[*]/', sql)
        return m and m.group(1) or ''
    def query(self, sql, locals=None):
        pystmt = self.parse_pystmt(sql)
        exec pystmt in self.env, locals
        src, tname = self.parse_load_sql(sql)
        if src:
            self.load(tname, load_file(src, self.env, locals))
            return [], []
        cursor = self.execute(sql)
        self.commit()
        if not cursor.description: return [], []
        return [col[0] for col in cursor.description], list(cursor)
    def queryp(self, sql, locals):
        self.before_execute(sql)
        header, rows = self.query(sql, locals)
        if rows:
            render_result(rows, header, os.getenv('term', 'text'))
    def load(self, tname, data): # data is iterator
        first_row = data.next()
        tschema = self.prepare_table(tname, first_row)
        self.executemany(tschema.replace_sql(), parse_rows(tschema.col_types, [first_row]))
        self.executemany(tschema.replace_sql(), parse_rows(tschema.col_types, data))
        self.commit()
    def prepare_table(self, tname, first_row):
        if re.match('\w+$', tname):
            return TSchema(self.get_tschema(tname))
        else:
            x = TSchema(tname or os.getenv('table_schema') or TSchema.guess_tschema(first_row))
            self.before_execute(x.create_sql())
            # self.execute(x.drop_sql())
            self.execute(x.create_sql())
            self.commit()
            return x
    def get_tschema(self, tname):
        tschema = list(self.execute('select sql from sqlite_master where name="%s"'%(tname)))
        if not tschema: raise QueryErr('no such table', tname)
        return tschema[0][0]
    def executemany(self, sql, data):
        return self.conn.executemany(sql, data)
    def execute(self, sql):
        return self.conn.execute(sql)
    def commit(self):
        self.conn.commit()

__doc__ += '''
Aggregate Function and Other Extension:
 - select * from order by key1 alphanum
 - wavg(weight, qty), std(qty)
 - first(x), last(x), str_join(x)
 - corrcoef(x, y)
 - correlate(x, y)
 - auto_correlate(x)
 - plot("file.png r+", "y", x, y) or plot("file.png r+", "", y)
 - hist("file.png n_bins", "", x)
 - corr(maxlags, "", x, y)
 - scatter(n_bins, "", x, y)
 - bar(x1,x2,x3...)
Sql Function:
 - Ts2Str(x)
 - HighPass(x, limit)
 - LowPass(x, limit)
 - Int(key, x)
 - Diff(key, x)
 - AssignId(key, x)
Multi Img Plot:
  select plot("a%%.png", "", y) from t1 group by type;
SuperImpose Plot:
  select plot("a.png-", "",  y) from t1 group by type union select plot("a.png")
Set figsize
  select /*py: figsize=(100,4) */plot("a.png-", "",  y) from t1
Set figsize
'''
import os
import re
import math
import sqlite3
import itertools
import time
def transpose(matrix):
    cols = [[] for i in matrix[0]] # Note: You can not write cols = [[]] * len(matrix[0]); if so, all col in cols will ref to same list object
    for row in matrix:
        map(lambda col,i: col.append(i), cols, row)
    return cols

def safe_int(x, default=0):
    try:
        return int(x)
    except ValueError:
        return default

def list_slices(seq, *slices):
    return [seq[i] for i in slices]

def dict_slice(d, *keys):
    return [d.get(x) for x in keys]

class SqliteAgg:
    def __init__(self, **kw):
        self.kw = kw
        self.data = []

    def step(self, *values):
        self.data.append(values)

    def finalize(self):
        return None

def make_sqlite_agg_class(func):
    class SqliteAggClass(SqliteAgg):
        def __init__(self):
            SqliteAgg.__init__(self)

        def finalize(self):
            return func(self.data)
    return SqliteAggClass

try:
    import numpy as np
except:
    np = None
def weighted_avg(x, w):
    return np.average(x, weights=w)
def correlate(x, y):
    return np.mean(np.correlate(x, y))
def auto_correlate(x):
    return np.mean(np.correlate(x, x))

def first(x):
    return x[0]

def last(x):
    return x[-1]

def str_join(x):
    return ','.join(map(str, x))

def NotZero(x):
    return x != 0

class IdGen:
    def __init__(self, key, init=0):
        self.key = key
        self.next_id = itertools.count(init)
        self.id_map = {}
    def get_id(self, x):
        if not self.id_map.has_key(x):
            id = self.next_id.next()
            self.id_map[x] = id
            if 0 == (id % 100000):
                sys.stderr.write("IdGen {} {}\n".format(self.key, id))
        return self.id_map[x]

class Intergal:
    def __init__(self, key, init=0):
        self.total = init
    def calc(self, x):
        self.total += x
        return self.total

class Differential:
    def __init__(self, key, init=0):
        self.last = init
    def calc(self, x):
        d = x - self.last
        self.last = x
        return d

class CacheTest:
    def __init__(self):
        self.cache = {}
        self.ref, self.load = 0, 0
    def step(self, limit, *value):
        self.access(limit, value)
    def finalize(self):
        return self.load
    def add(self, limit, x):
        self.cache[x] = self.new_ref(x)
        if len(self.cache) > limit:
            x = self.evict()
            del self.cache[x]
    def ref_exist(self, x): pass
    def access(self, limit, x):
        self.ref += 1
        if x not in self.cache:
            self.add(limit, x)
            self.load += 1
        else:
            self.ref_exist(self.cache[x])
        if 0 == (self.ref % 100000):
            sys.stderr.write("cache stat: ref={} load={}\n".format(self.ref, self.load))

import Queue
class FifoCacheTest(CacheTest):
    def __init__(self):
        self.queue = Queue.Queue()
        CacheTest.__init__(self)
    def new_ref(self, x):
        self.queue.put(x)
        return 1
    def evict(self):
        return self.queue.get()

class DLinkNode:
    def __init__(self, key):
        self.key = key
        self.prev = self
        self.next = self
    def insert(self, n):
        prev = self
        next = self.next
        n.prev = prev
        n.next = next
        prev.next = n
        next.prev = n
        return n
    def delete(self):
        prev = self.prev
        next = self.next
        prev.next = next
        next.prev = prev
        return self

class LruCacheTest(CacheTest):
    def __init__(self):
        self.head = DLinkNode(None)
        CacheTest.__init__(self)
    def new_ref(self, x):
        return self.head.insert(DLinkNode(x))
    def ref_exist(self, x):
        self.head.insert(x.delete())
    def evict(self):
        return self.head.prev.delete().key

global_sql_vars = {}
def get_global_sql_func(key, builder):
    if key not in global_sql_vars:
        global_sql_vars[key] = builder(key)
    return global_sql_vars[key]

def Int(key, x):
    return get_global_sql_func(key, Intergal).calc(x)

def Diff(key, x):
    return get_global_sql_func(key, Differential).calc(x)

def AssignId(key, x):
    return get_global_sql_func(key, IdGen).get_id(x)

def HighPass(x, limit):
    if x >  limit:
        return x
    else:
        return 0

def LowPass(x, limit):
    if x < limit:
        return x
    else:
        return 0

def Log(x, base):
    if x == 0:
        return 0
    else:
        return math.log(x, base)

def Ts2Str(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(ts * 1000000)/1e6)) + '.%06d'%(int(ts * 1000000) % 1000000)

def resub(s, f, t):
    return re.sub(f, t, s)

def awk(s, n, m=0):
    if m == 0:
        return s.split()[n]
    return ' '.join(s.split()[n:m])
def pypipe(s, c):
    return eval(c)(s)

def uint64_shift(x): return int(x) - (1<<63)

SqliteStd = make_sqlite_agg_class(lambda data:np.std(transpose(data)[0]))
SqliteCorrelate = make_sqlite_agg_class(lambda data:correlate(*transpose(data)))
SqliteAutoCorrelate = make_sqlite_agg_class(lambda data:auto_correlate(transpose(data)[0]))
SqliteCorrcoef = make_sqlite_agg_class(lambda data:np.corrcoef(*transpose(data))[0][1])
SqliteWeightedAvg = make_sqlite_agg_class(lambda data:weighted_avg(*transpose(data)))
SqliteFirst = make_sqlite_agg_class(lambda data:first(data)[0])
SqliteLast = make_sqlite_agg_class(lambda data:last(data)[0])
SqliteStrJoin = make_sqlite_agg_class(lambda data:str_join(transpose(data)[0]))

try:
    import matplotlib
    import matplotlib.dates as mdates
except:
    matplotlib = None
    # print 'not support matplotlib'

if matplotlib:
    matplotlib.use('Agg')
    #matplotlib.use('Cairo')
    #matplotlib.rcParams['figure.figsize']=(8,6)
    matplotlib.rcParams['figure.figsize']=(20,4)
    import matplotlib.pyplot as plt

figsize = (20,4)
fignums = dict()

fignum_gen = itertools.count(0)
def gen_fig_num(path):
    if path not in fignums:
        fignums[path] = fignum_gen.next()
    return fignums[path]

id_gen = itertools.count(0)
def gen_fig_path(path):
    return re.sub('%', '%02d'%(id_gen.next()), path)

def render_img_result(path):
    if os.getenv('term', 'text') == 'html':
        return '<img src="%s%s" alt="%s"/>' % (http_img_root, path, path)
    else:
        return path

def make_plot_func(func):
    if not matplotlib: return lambda data: "not support"
    def parse_matplot_spec(spec):
        spec = spec.split(' ', 1)
        if len(spec) == 1: return spec[0], ''
        else: return spec

    def plot(data):
        if (not data) or (not data[-1]) : return None
        cols = transpose(data)
        spec, cols = cols[0][-1], cols[1:]
        path, args = parse_matplot_spec(spec)
        to_be_cont = path.endswith('-')
        path = gen_fig_path(re.sub('-$', '', path))
        num = gen_fig_num(path)
        if cols:
            label, cols = cols[0][-1], cols[1:]
            plt.figure(num=num, figsize=figsize, frameon=True, tight_layout=True)
        #cols = [map(float, col) for col in cols]
            func(label, args, *cols)
        if to_be_cont:
            return 'impose on %s'%(path,)
        elif not path:
            plt.legend(loc='upper right')
            plt.tight_layout(0)
            plt.show()
        else:
            plt.legend(loc='upper right')
            plt.savefig(path)
            return render_img_result(path)
    return plot

def make_sqlite_plot_func(func):
    return make_sqlite_agg_class(make_plot_func(func))

def plot(label, args, x, y=None):
    if y == None:
        plt.plot(x, args, label=label)
    elif type(x[0]) != unicode and type(x[0]) != str:
        plt.plot(x, y, args, label=label)
    elif TSchema.guess_type(x[0]) == 'datetime':
        plt.plot([parse_cell('datetime', i) for i in x], y, args, label=label)
        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%H:%M:%S')
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.gca().xaxis_date()
    else:
        plt.xticks(np.arange(len(x)), x, rotation='-15')
        plt.plot(y, label=label)

def hist(label, args, x):
    plt.hist(x, bins=safe_int(args,10), label=label)

def scatter(label, args, x, y):
    hist2d = np.histogram2d(y, x, bins=safe_int(args,10))[0]
    plt.imshow(hist2d)

def corr(label, args, x, y):
    plt.ylim(0, 1)
    plt.xcorr(x, y, maxlags=safe_int(args, None), label=label)

def acorr(label, args, x):
    plt.ylim(0, 1)
    plt.xcorr(x, y, maxlags=safe_int(args, None), label=label)

def bar(label, args, *x):
    if not x: return None
    w = 0.8/len(x)
    colors = 'bgrcmy'
    groupX = np.array(range(len(x[0])))
    for i,col in enumerate(x):
        plt.bar(groupX+i*w, col, w, color=colors[i%len(colors)], label=label)

SqlitePlot = make_sqlite_plot_func(plot)
SqliteHist = make_sqlite_plot_func(hist)
SqliteScatter = make_sqlite_plot_func(scatter)
SqliteCorr = make_sqlite_plot_func(corr)
SqliteBar = make_sqlite_plot_func(bar)

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def extend(conn):
    conn.create_function("resub", 3, resub)
    conn.create_function("awk", 2, awk)
    conn.create_function("awk", 3, awk)
    conn.create_function("pipe", 2, pypipe)
    conn.create_function("uint64_shift", 1, uint64_shift)
    conn.create_function("Ts2Str", 1, Ts2Str)
    conn.create_function("NotZero", 1, NotZero)
    conn.create_function("HighPass", 2, HighPass)
    conn.create_function("LowPass", 2, LowPass)
    conn.create_function("AssignId", 2, AssignId)
    conn.create_function("AssignId", 3, AssignId)
    conn.create_function("Int", 2, Int)
    conn.create_function("Diff", 2, Diff)
    conn.create_function("log", 2, Log)
    conn.create_collation("alphanum", lambda x1,x2: cmp(alphanum_key(x1), alphanum_key(x2)))
    conn.create_aggregate("wavg", 2, SqliteWeightedAvg)
    conn.create_aggregate("std", 1, SqliteStd)
    conn.create_aggregate("corrcoef", 2, SqliteCorrcoef)
    conn.create_aggregate("correlate", 2, SqliteCorrelate)
    conn.create_aggregate("auto_correlate", 1, SqliteAutoCorrelate)
    conn.create_aggregate("first", 1, SqliteFirst)
    conn.create_aggregate("last", 1, SqliteLast)
    conn.create_aggregate("str_join", 1, SqliteStrJoin)
    conn.create_aggregate("plot", 1, SqlitePlot)
    conn.create_aggregate("plot", 3, SqlitePlot)
    conn.create_aggregate("plot", 4, SqlitePlot)
    conn.create_aggregate("hist", 3, SqliteHist)
    conn.create_aggregate("corr", 4, SqliteCorr)
    conn.create_aggregate("scatter", 4, SqliteScatter)
    conn.create_aggregate("fifo_cache_test", 2, FifoCacheTest)
    conn.create_aggregate("lru_cache_test", 2, LruCacheTest)
    for i in range(2,32):
        conn.create_aggregate("bar", i, SqliteBar)
    return conn

def help():
    print sys.argv
    print __doc__

if __name__ == '__main__':
    (len(sys.argv) == 2 or len(sys.argv) == 1 and not sys.stdin.isatty()) or help() or sys.exit(1)
    if len(sys.argv) == 2:
        sql = sys.argv[1]
    else:
        sql = sys.stdin.read()
    msql = re.split(';|\n', sql)
    db_path = os.getenv('db_path') or ':memory:'
    conn = TConn(db_path, globals())
    first_cmd = msql[0].strip()
    if db_path == ':memory:' and not (first_cmd.startswith('load ') or first_cmd.startswith('create')):
        msql.insert(0, 'load stdin')
    for sql in msql:
        if sql.strip():
            conn.queryp(sql.strip(), None)
