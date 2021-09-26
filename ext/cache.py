'''
FifoCacheTest(limit, x)
LruCacheTest(limit, x)
'''
class CacheTest:
    def __init__(self):
        self.cache = {}
        self.ref, self.load = 0, 0
    def step(self, limit, *value):
        self.access(limit, value)
    def finalize(self):
        return self.load
    def add(self, limit, x):
        self.cache[x] = self.new_ref(x)
        if len(self.cache) > limit:
            x = self.evict()
            del self.cache[x]
    def ref_exist(self, x): pass
    def access(self, limit, x):
        self.ref += 1
        if x not in self.cache:
            self.add(limit, x)
            self.load += 1
        else:
            self.ref_exist(self.cache[x])
        if 0 == (self.ref % 100000):
            sys.stderr.write("cache stat: ref={} load={}\n".format(self.ref, self.load))

import Queue
class FifoCacheTest(CacheTest):
    def __init__(self):
        self.queue = Queue.Queue()
        CacheTest.__init__(self)
    def new_ref(self, x):
        self.queue.put(x)
        return 1
    def evict(self):
        return self.queue.get()

class DLinkNode:
    def __init__(self, key):
        self.key = key
        self.prev = self
        self.next = self
    def insert(self, n):
        prev = self
        next = self.next
        n.prev = prev
        n.next = next
        prev.next = n
        next.prev = n
        return n
    def delete(self):
        prev = self.prev
        next = self.next
        prev.next = next
        next.prev = prev
        return self

class LruCacheTest(CacheTest):
    def __init__(self):
        self.head = DLinkNode(None)
        CacheTest.__init__(self)
    def new_ref(self, x):
        return self.head.insert(DLinkNode(x))
    def ref_exist(self, x):
        self.head.insert(x.delete())
    def evict(self):
        return self.head.prev.delete().key

def reg_sqlite_ext(self, conn):
    self.reg_aggregate_class(conn, FifoCacheTest, 2)
    self.reg_aggregate_class(conn, LruCacheTest, 2)
