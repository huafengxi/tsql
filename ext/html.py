'''
HtmlLink(url, text=None)
'''
def HtmlLink(url, text=None):
    return '<a href="%s" target="_blank">%s<a>'%(url, text)

def reg_sqlite_ext(self, conn):
    self.reg_func(conn, HtmlLink, 1)
    self.reg_func(conn, HtmlLink, 2)
