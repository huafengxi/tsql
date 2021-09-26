'''
IdGen(key, x)
Integral(key, x)
Diff(key, x)
HighPass(x, limit)
LowPass(x, limit)
Log(x, base)
NotZero(x)
'''
import itertools
import sys
class IdGen:
    def __init__(self, key, init=0):
        self.key = key
        self.next_id = itertools.count(init)
        self.id_map = {}
    def __call__(self, x):
        if not self.id_map.has_key(x):
            id = self.next_id.next()
            self.id_map[x] = id
            if 0 == (id % 100000):
                sys.stderr.write("IdGen {} {}\n".format(self.key, id))
        return self.id_map[x]

class Integral:
    def __init__(self, key, init=0):
        self.total = init
    def __call__(self, x):
        self.total += x
        return self.total

class Diff:
    def __init__(self, key, init=0):
        self.last = init
    def __call__(self, x):
        d = x - self.last
        self.last = x
        return d

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

def NotZero(x):
    return x != 0

def reg_sqlite_ext(self, conn):
    self.reg_key_func(conn, Integral, 2)
    self.reg_key_func(conn, Diff, 2)
    self.reg_key_func(conn, IdGen, 2)
    self.reg_func(conn, HighPass, 2)
    self.reg_func(conn, LowPass, 2)
    self.reg_func(conn, Log, 2)
    self.reg_func(conn, NotZero, 1)
