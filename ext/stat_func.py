'''
wavg(x, w)
correlate(x, y)
auto_correlate(x)
first(x)
last(x)
'''
try:
    import numpy as np
except:
    np = None
def wavg(x, w):
    return np.average(x, weights=w)
def correlate(x, y):
    return np.mean(np.correlate(x, y))
def auto_correlate(x):
    return np.mean(np.correlate(x, x))

def first(x):
    return x[0]

def last(x):
    return x[-1]

def reg_sqlite_ext(self, conn):
    self.reg_aggregate_func(conn, wavg, 2)
    self.reg_aggregate_func(conn, correlate, 2)
    self.reg_aggregate_func(conn, auto_correlate, 1)
    self.reg_aggregate_func(conn, first, 1)
    self.reg_aggregate_func(conn, last, 1)
