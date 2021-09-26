'''
plot("file.png r+", "y", x, y) or plot("file.png r+", "", y)
hist("file.png n_bins", "", x)
corr(maxlags, "", x, y)
scatter(n_bins, "", x, y)
bar(x1,x2,x3...)
multi img plot:
  select plot("a%%.png", "", y) from t1 group by type;
superimpose plot:
  select plot("a.png-", "",  y) from t1 group by type union select plot("a.png")
set figsize
  select /*py: figsize=(100,4) */plot("a.png-", "",  y) from t1
'''
try:
    import matplotlib
    import matplotlib.dates as mdates
except:
    matplotlib = None
    print 'not support matplotlib'

if matplotlib:
    matplotlib.use('Agg')
    #matplotlib.use('Cairo')
    #matplotlib.rcParams['figure.figsize']=(8,6)
    matplotlib.rcParams['figure.figsize']=(20,4)
    import matplotlib.pyplot as plt

figsize = (20,4)
fignums = dict()

import itertools
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

@make_plot_func
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

@make_plot_func
def hist(label, args, x):
    plt.hist(x, bins=safe_int(args,10), label=label)

@make_plot_func
def scatter(label, args, x, y):
    hist2d = np.histogram2d(y, x, bins=safe_int(args,10))[0]
    plt.imshow(hist2d)

@make_plot_func
def corr(label, args, x, y):
    plt.ylim(0, 1)
    plt.xcorr(x, y, maxlags=safe_int(args, None), label=label)

@make_plot_func
def acorr(label, args, x):
    plt.ylim(0, 1)
    plt.xcorr(x, y, maxlags=safe_int(args, None), label=label)

@make_plot_func
def bar(label, args, *x):
    if not x: return None
    w = 0.8/len(x)
    colors = 'bgrcmy'
    groupX = np.array(range(len(x[0])))
    for i,col in enumerate(x):
        plt.bar(groupX+i*w, col, w, color=colors[i%len(colors)], label=label)

def reg_sqlite_ext(self, conn):
    self.reg_aggregate_func(conn, plot, 1)
    self.reg_aggregate_func(conn, plot, 3)
    self.reg_aggregate_func(conn, plot, 4)
    self.reg_aggregate_func(conn, hist, 3)
    self.reg_aggregate_func(conn, corr, 4)
    self.reg_aggregate_func(conn, scatter, 4)
    for i in range(2,32):
        self.reg_aggregate_func(conn, bar, i)
