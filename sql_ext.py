class SqlHook:
    def __init__(self):
        self.doc, self.hook = [], []
    def load_exts(self, list):
        for m in list.split():
            if m.startswith('#'): continue
            self.load_ext(m)
    def load_ext(self, name):
        m = __import__(name)
        self.doc.append('* %s'%(name) + m.__doc__)
        self.hook.append(m.reg_sqlite_ext)
    def get_doc(self):
        return '\n'.join(self.doc)
    def add_doc(self, d):
        self.doc.append(d)
    def extend(self, conn):
        for h in self.hook:
            h(self, conn)
        return conn

    def reg_aggregate_func(self, conn, func, args_num):
        cls = make_sqlite_agg_class(func, args_num)
        conn.create_aggregate(func.__name__, args_num, cls)
    def reg_aggregate_class(self, conn, cls, args_num):
        conn.create_aggregate(cls.__name__, args_num, cls)
    def reg_func(self, conn, func, args_num):
        conn.create_function(func.__name__, args_num, func)
    def reg_key_func(self, conn, cls, args_num):
        def func(key, *args):
            return get_global_sql_func(key, cls)(*args)
        conn.create_function(cls.__name__, args_num, func)

def transpose(matrix):
    cols = [[] for i in matrix[0]] # Note: You can not write cols = [[]] * len(matrix[0]); if so, all col in cols will ref to same list object
    for row in matrix:
        map(lambda col,i: col.append(i), cols, row)
    return cols

def make_sqlite_agg_class(func, args_num):
    class SqliteAggClass:
        def __init__(self, **kw):
            self.kw = kw
            self.data = []
        def step(self, *values):
            self.data.append(values)
        def finalize(self):
            args = transpose(self.data)[:args_num]
            return func(*args)
    return SqliteAggClass

global_sql_vars = {}
def get_global_sql_func(key, builder):
    if key not in global_sql_vars:
        global_sql_vars[key] = builder(key)
    return global_sql_vars[key]
