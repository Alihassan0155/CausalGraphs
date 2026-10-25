"""Bounded, collision-free Fisher-Z memoization; upstream numerical code is unchanged."""
from collections import OrderedDict
from collections.abc import MutableMapping
import struct
import sys

from causallearn.utils.cit import FisherZ, register_ci_test


class BoundedCache(MutableMapping):
    """Metadata is never evicted. CI values use a bounded LRU (zero disables memoization)."""
    def __init__(self, metadata, limit):
        self.metadata = dict(metadata)
        self.values = OrderedDict()
        self.limit = limit
        self.lookups = self.hits = self.insertions = self.evictions = self.high_water = 0

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.metadata
        self.lookups += 1
        hit = key in self.values
        self.hits += int(hit)
        return hit

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.metadata[key]
        value = self.values[key]
        self.values.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.metadata[key] = value
            return
        self.insertions += 1
        if not self.limit:
            return
        self.values[key] = value
        self.values.move_to_end(key)
        if len(self.values) > self.limit:
            self.values.popitem(last=False)
            self.evictions += 1
        self.high_water = max(self.high_water, len(self.values))

    def __delitem__(self, key):
        del (self.metadata if isinstance(key, str) else self.values)[key]

    def __iter__(self):
        yield from self.metadata
        yield from self.values

    def __len__(self):
        return len(self.metadata) + len(self.values)

    def statistics(self):
        # Exact shallow Python object sizes for live keys/values and the mapping;
        # excludes allocator fragmentation, input arrays and temporary objects.
        size = sys.getsizeof(self.values) + sum(sys.getsizeof(k) + sys.getsizeof(v)
                                                for k, v in self.values.items())
        return dict(entries=len(self.values), limit=self.limit, lookups=self.lookups,
                    hits=self.hits, insertions=self.insertions, evictions=self.evictions,
                    high_water=self.high_water, live_cache_object_bytes=size)


class MemoryFisherZ(FisherZ):
    def __init__(self, data, cache_entries=100000, **kwargs):
        if kwargs.get('cache_path') is not None:
            raise ValueError('Compact keys use runner pickle checkpoints, not upstream JSON caches.')
        if not 0 <= cache_entries or data.shape[1] > 65536:
            raise ValueError('Nonnegative cache limit and <=65536 variables required.')
        super().__init__(data, **kwargs)
        self.pvalue_cache = BoundedCache(self.pvalue_cache, cache_entries)

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        # Same canonicalization as CIT_Base for scalar Fisher-Z indices. Fixed-width
        # unsigned 16-bit encoding is injective, including conditioning-set length.
        condition_set = sorted(set(map(int, () if condition_set is None else condition_set)))
        X, Y = (int(X), int(Y)) if X < Y else (int(Y), int(X))
        assert X not in condition_set and Y not in condition_set, 'X, Y cannot be in condition_set.'
        key = struct.pack('<' + 'H' * (2 + len(condition_set)), X, Y, *condition_set)
        return [X], [Y], condition_set, key

    # Intentionally inherit FisherZ.__call__ without copying/changing ANY arithmetic.


register_ci_test('memory_fisherz', MemoryFisherZ)
