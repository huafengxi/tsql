#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-
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
        if type(x) == str or type(x) == str or not hasattr(x, '__getitem__'):
            return (x,)
        return x
    #if path.startswith('@'):
    #    return [to_row(i) for i in eval(path[1:], globals, locals)]
    if path == 'stdin':
        fd = sys.stdin
    elif path.startswith('!'):
        fd = os.popen(path[1:])
    else:
        fd = open(path)
    for line in fd:
        if not line.strip(): continue
        yield re.split(col_sep, line.strip())

def parse_cell(t, cell):
    def safe_float(x):
        if x == 'null' or x == '':
            return None
        try:
            return float(x)
        except TypeError as ValueError:
            return None
    def safe_int(x):
        if type(x) == str:
            x = x.split('.')[0]
        if x == 'null' or x == '':
            return None
        try:
            if x.startswith('0x'): return int(x, 16)
            return int(x)
        except TypeError as ValueError:
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
        return text.decode('utf8') if text is bytes else text
    value_parsers = dict(real=safe_float, text=to_unicode, integer=safe_int, bigint=safe_int, int=safe_int, datetime=to_datetime, boolean=bool)
    if t not in value_parsers:
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
        yield list(map(lambda type, cell: parse_cell(type, cell), types, row))

def render_result(result, header, term):
    def to_str(o):
        if type(o) == str:
            return o.encode('utf8')
        else:
            return str(o)
    def render_row(cols):
        return '<tr>%s</tr>'%(''.join('<td>%s</td>'%(to_str(cell)) for cell in cols))
    if len(result) <= 0:
        sys.stderr.write(' empty set.')
    else:
        if term == 'html':
            print('<table>')
            print(render_row(header))
            for cols in result:
                print(render_row(cols))
            print('</table>')
        else:
            sys.stderr.write('%s\n'%('\t'.join(header)))
            for cols in result:
                print('\t'.join(map(to_str, cols)))

from tschema import TSchema
class TConn:
    def __init__(self, path, env):
        sqlite3.enable_callback_tracebacks(True)
        self.conn = sql_hook.extend(sqlite3.connect(path))
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
        exec(pystmt, self.env, locals)
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
        first_row = next(data)
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

_tsql_path_ = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append('%s/ext'%(_tsql_path_))
from sql_ext import SqlHook
sql_hook = SqlHook()
sql_hook.load_exts('pypipe stat_func filt plot str')

def help():
    print(sys.argv)
    print(__doc__)
    print(sql_hook.get_doc())

if __name__ == '__main__':
    (len(sys.argv) == 2 or len(sys.argv) == 1 and not sys.stdin.isatty()) or help() or sys.exit(1)
    if len(sys.argv) == 2:
        sql = sys.argv[1]
    else:
        sql = sys.stdin.read()
    msql = re.split(';|\n', sql)
    db_path = os.getenv('db_path') or ':memory:'
    if os.path.exists('tsql_init.py'):
        exec(compile(open('tsql_init.py').read(), 'tsql_init.py', 'exec'), globals(), locals())
    conn = TConn(db_path, globals())
    first_cmd = msql[0].strip()
    if db_path == ':memory:' and not (first_cmd.startswith('load ') or first_cmd.startswith('create')):
        msql.insert(0, 'load stdin')
    for sql in msql:
        if sql.strip():
            conn.queryp(sql.strip(), None)
