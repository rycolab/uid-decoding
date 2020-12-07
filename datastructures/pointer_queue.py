from sortedcontainers import SortedDict

class PointerQueue(object):
    """
    Implementation of linked list style queue
    """
    def __init__(self, initial, reserve=0):
        # Sorted dict has O(log(n)) runtime for pop, set, and delete operations
        # May implement faster cython DS in future
        self.queue = SortedDict(zip(initial, range(len(initial))))
        # Key for looking up queues from specific time steps
        self.pointer = initial
        self.pointer.extend([None]*reserve)

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    def __list__(self):
        return self.pointer
    
    def __setitem__(self, key, value):
        self.queue[key] = value
        self.pointer[value] = key

    def __next__(self):
        try:
            return self.queue.popitem()
        except AssertionError:
            raise StopIteration

    def __nonzero__(self):
        return self.queue

    def pop(self, key, **kwargs):
        return self.queue.pop(key, **kwargs)

    def popindex(self, index, **kwargs):
        return self.queue.pop(self.pointer[index], **kwargs)