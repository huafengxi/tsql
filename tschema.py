import re
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
        m = re.search('([_0-9a-zA-Z]+) *\(([^)]+)\)$', table_schema.strip())
        if not m:
            raise QueryErr('table_schema syntax error[%s]'%(table_schema))
        else:
            table_name, col_schemas = m.groups()
        return table_name, [col.strip().split(' ', 2)[:2] for col in col_schemas.split(',')]
