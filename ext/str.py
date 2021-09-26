'''
str_join(x)
Ts2Str(ts)
resub(str, from, to)
awk(str, start, end=-1)
select * from order by key1 alphanum
'''
import re
import time
def Ts2Str(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(ts * 1000000)/1e6)) + '.%06d'%(int(ts * 1000000) % 1000000)

def resub(s, f, t):
    return re.sub(f, t, s)

def str_join(x):
    return ','.join(map(str, x))

def awk(s, n, m=0):
    if m == 0:
        return s.split()[n]
    return ' '.join(s.split()[n:m])

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

def reg_sqlite_ext(self, conn):
    self.reg_aggregate_func(conn, str_join, 1)
    self.reg_func(conn, Ts2Str, 1)
    self.reg_func(conn, resub, 3)
    self.reg_func(conn, awk, 2)
    self.reg_func(conn, awk, 3)
    conn.create_collation("alphanum", lambda x1,x2: cmp(alphanum_key(x1), alphanum_key(x2)))
