"""CPU-only GES helper optimization; same operators, scores and iteration order."""
import discover as b
from ges_recovery import build as reference_build


def insert_vc2_fast(j,i,NAT_set,semi_succ):
    # Original predicate: no j->i semi-directed path whose internal/start
    # vertices avoid NAT. A visited DFS suffices; enumerating every simple
    # path, as upstream does, is unnecessary for a Boolean reachability result.
    if j==i:return 0
    if j in NAT_set:return 1
    pending=[j];visited={j}
    while pending:
        node=pending.pop()
        for child in semi_succ.get(node,()):
            if child==i:return 0
            if child not in NAT_set and child not in visited:
                visited.add(child);pending.append(child)
    return 1


def find_subset_include(s0,sub):
    if len(s0)==0 or len(sub)==0:return b.np.ones(len(sub))
    required=set(s0)
    return b.np.fromiter((required.issubset(candidate) for candidate in sub),dtype=float,count=len(sub))


def build(*args,**kwargs):
    function=reference_build(*args,**kwargs)
    # The GES function looks up these imported helpers in its own copied globals.
    # No installed modules or reference functions are modified.
    function.__globals__.update(insert_vc2_fast=insert_vc2_fast,find_subset_include=find_subset_include)
    return function
