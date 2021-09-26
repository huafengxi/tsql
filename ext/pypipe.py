'''
pypipe(x, pyexpr)
'''
def pypipe(s, c):
    return eval(c)(s)

def reg_sqlite_ext(self, conn):
    self.reg_func(conn, pypipe, 2)
