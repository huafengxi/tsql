'''
HtmlLink(url, text=None)
'''
def HtmlLink(url, text=None):
    return '<a href="%s" target="_blank">%s<a>'%(url, text)
def ParamLink(tpl, arg):
    return HtmlLink(tpl % arg, arg)

def ImgLink(tpl, arg):
    return '<img src="%s" alt="%s"/>'%(tpl % arg, arg)

def reg_sqlite_ext(self, conn):
    self.reg_func(conn, HtmlLink, 1)
    self.reg_func(conn, HtmlLink, 2)
    self.reg_func(conn, ParamLink, 2)
    self.reg_func(conn, ImgLink, 2)
