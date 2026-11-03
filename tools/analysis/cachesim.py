"""Expert cache and read-ahead policies replayed on a routing trace from myds4.

  build/myds4 -m CHECKPOINT -p PROMPT -n TOKENS --routes-fd 3 3>TRACE
  python tools/analysis/cachesim.py TRACE [--front MS --expert MS] SLOTS...

For every decoded token and layer, the trace holds the 6 selected experts,
three predictions, each the top 16 of a router applied to a row known
earlier, and the router's 384 logits (not used here):

  p1, p2  the next two layers' routers applied to this layer's FFN input
  pc      this layer, from the residual part of its FFN input, known before
          its attention

The trace keeps the native router forecasts even when online correction is
enabled. It does not record corrected prefetch candidates or their read timing.
The first 28 decoded tokens warm the cache; the rest are counted.

Demand-only policies: least recently used (LRU), least frequently used (LFU,
recency breaking ties), LRU-2, the adaptive replacement cache (ARC),
segmented LRU (SLRU), Belady's optimal policy, and myds4's scores: an
expert's score sums a weight for each token that selected it, the weight
doubling every HALF tokens; the lowest score is evicted.

Read-ahead policies replay decoding with a timing model, evicting by score:
a layer's front (attention, router) takes FRONT and its experts' work EXPERT
milliseconds (0.6 and 0.3 unless given: 36 ms per token); the
SSD reads one expert at a time, READ seconds each, demand reads before
queued read-aheads; a layer's experts start when its experts are in. At
START into its front, pc may read ahead for the layer; at its routing, the
layer's selected experts that are not cached are read on demand, and p1 and
p2 read ahead for the next two layers. A source reads its top `k` that are
not cached (at most `admit` of them); with refinement it first cancels the
queued, unstarted read-aheads for its layer outside its top KEEP; with
demotion, read-aheads that their layer did not select move to the eviction
end of the LRU list. Oracles read ahead the actual experts of the layer a
given number of layers on (LEADS), across tokens.
"""

import heapq
import sys
from collections import OrderedDict, defaultdict

import numpy as np

LAYERS, EXPERTS, TOPK, PREDICT = 40, 384, 6, 16
NAMES = ("p1", "p2", "pc")
RECORD = TOPK + len(NAMES) * PREDICT + EXPERTS  # 4-byte words per layer and token: ids, then logits
LEADS = (2, 4, 8, 40)  # the oracles' layers ahead
WARM = 28
FRONT, EXPERT, READ, START = 0.6e-3, 0.3e-3, 1.33e-3, 0.15e-3
KEEP = 6
HALF = 50


def demand_requests(trace):
    return [l * EXPERTS + e for tok in trace for l, rec in enumerate(tok) for e in rec["act"]]


def lru(reqs, cap, warm):
    c, miss = OrderedDict(), 0
    for i, r in enumerate(reqs):
        if r in c:
            c.move_to_end(r)
        else:
            miss += i >= warm
            c[r] = 1
            if len(c) > cap:
                c.popitem(last=False)
    return miss


def lfu(reqs, cap, warm):
    count, c, miss, heap = defaultdict(int), {}, 0, []
    for i, r in enumerate(reqs):
        count[r] += 1
        if r not in c:
            miss += i >= warm
            if len(c) >= cap:
                while True:
                    n, t, v = heapq.heappop(heap)
                    if v in c and c[v] == (n, t):
                        del c[v]
                        break
        c[r] = (count[r], i)
        heapq.heappush(heap, (count[r], i, r))
    return miss


def lru2(reqs, cap, warm):
    """Evicts the expert whose second-to-last use is oldest (never used twice: oldest last use first)."""
    hist, c, miss, heap = {}, set(), 0, []
    for i, r in enumerate(reqs):
        h = hist.get(r, (-1, -1))
        h = (h[1], i)
        hist[r] = h
        if r not in c:
            miss += i >= warm
            if len(c) >= cap:
                while True:
                    k2, k1, v = heapq.heappop(heap)
                    if v in c and hist[v] == (k2, k1):
                        c.discard(v)
                        break
            c.add(r)
        heapq.heappush(heap, (h[0], h[1], r))
    return miss


def arc(reqs, cap, warm):
    """Adaptive replacement cache (Megiddo and Modha)."""
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p, miss = 0, 0

    def replace(r):
        if t1 and (len(t1) > p or (r in b2 and len(t1) == p)):
            k, _ = t1.popitem(last=False)
            b1[k] = 1
        else:
            k, _ = t2.popitem(last=False)
            b2[k] = 1

    for i, r in enumerate(reqs):
        if r in t1:
            del t1[r]
            t2[r] = 1
        elif r in t2:
            t2.move_to_end(r)
        else:
            miss += i >= warm
            if r in b1:
                p = min(cap, p + max(len(b2) // max(len(b1), 1), 1))
                replace(r)
                del b1[r]
                t2[r] = 1
            elif r in b2:
                p = max(0, p - max(len(b1) // max(len(b2), 1), 1))
                replace(r)
                del b2[r]
                t2[r] = 1
            else:
                if len(t1) + len(b1) == cap:
                    if len(t1) < cap:
                        b1.popitem(last=False)
                        replace(r)
                    else:
                        t1.popitem(last=False)
                elif len(t1) + len(b1) < cap and len(t1) + len(t2) + len(b1) + len(b2) >= cap:
                    if len(t1) + len(t2) + len(b1) + len(b2) == 2 * cap:
                        b2.popitem(last=False)
                    replace(r)
                t1[r] = 1
    return miss


def slru(reqs, cap, warm, protect=0.8):
    """Segmented LRU: a new expert enters a probation list; a second use moves it to a protected
    list of at most `protect` x cap, whose least recent overflow goes back to probation. Evicts
    the least recent of probation (of protected when probation is empty)."""
    prob, prot, miss = OrderedDict(), OrderedDict(), 0
    for i, r in enumerate(reqs):
        if r in prot:
            prot.move_to_end(r)
        elif r in prob:
            del prob[r]
            prot[r] = 1
            if len(prot) > protect * cap:
                k, _ = prot.popitem(last=False)
                prob[k] = 1
        else:
            miss += i >= warm
            prob[r] = 1
            if len(prob) + len(prot) > cap:
                (prob if prob else prot).popitem(last=False)
    return miss


def belady(reqs, cap, warm):
    nxt, last = [0] * len(reqs), {}
    for i in range(len(reqs) - 1, -1, -1):
        nxt[i] = last.get(reqs[i], len(reqs) + i)
        last[reqs[i]] = i
    c, heap, miss = {}, [], 0
    for i, r in enumerate(reqs):
        if r not in c:
            miss += i >= warm
            if len(c) >= cap:
                while True:
                    n, v = heapq.heappop(heap)
                    if c.get(v) == -n:
                        del c[v]
                        break
        c[r] = nxt[i]
        heapq.heappush(heap, (-nxt[i], r))
    return miss


def scores(trace, cap, warm, half=HALF):
    """Demand misses when evicting the lowest score; warm counts tokens."""
    grow = 2 ** (1 / half)  # the weight of token t is grow**t
    s, cached, miss, n = np.zeros(LAYERS * EXPERTS), np.zeros(LAYERS * EXPERTS, bool), 0, 0
    for t, tok in enumerate(trace):
        for l, rec in enumerate(tok):
            s[l * EXPERTS + np.array(rec["act"])] += grow ** t
            for e in rec["act"]:
                key = l * EXPERTS + e
                if cached[key]:
                    continue
                miss += t >= warm
                if n >= cap:
                    v = np.where(cached, s, np.inf)
                    v[l * EXPERTS + np.array(rec["act"])] = np.inf  # not the layer's own experts
                    cached[v.argmin()] = False
                    n -= 1
                cached[key] = True
                n += 1
    return miss


def recall(trace, name, ahead, k):
    """Share of the target layer's selected experts among a prediction's top k."""
    hit = total = 0
    for tok in trace[WARM:]:
        for l in range(LAYERS - ahead):
            act = set(tok[l + ahead]["act"])
            hit += len(act & set(tok[l][name][:k]))
            total += len(act)
    return hit / total


class LRU:
    """The cache's eviction order: least recently used first."""

    def __init__(self, cap):
        self.cap, self.c = cap, OrderedDict()

    def __contains__(self, key):
        return key in self.c

    def __len__(self):
        return len(self.c)

    def use(self, key):  # a demand hit
        self.c.move_to_end(key)

    def add(self, key, demand):  # a demand miss, or a read-ahead
        self.c[key] = 1

    def demote(self, key):
        self.c.move_to_end(key, last=False)

    def remove(self, key):
        del self.c[key]

    def evict(self):
        return self.c.popitem(last=False)[0]


class ARC(LRU):
    """The adaptive replacement cache with read-aheads entering the recency list without
    adapting it, and demotion to that list's eviction end."""

    def __init__(self, cap):
        self.cap, self.t1, self.t2, self.b1, self.b2, self.p = cap, OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), 0

    def __contains__(self, key):
        return key in self.t1 or key in self.t2

    def __len__(self):
        return len(self.t1) + len(self.t2)

    def use(self, key):
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = 1
        else:
            self.t2.move_to_end(key)

    def add(self, key, demand):
        if demand and key in self.b1:
            self.p = min(self.cap, self.p + max(len(self.b2) // max(len(self.b1), 1), 1))
            del self.b1[key]
            self.t2[key] = 1
        elif demand and key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) // max(len(self.b2), 1), 1))
            del self.b2[key]
            self.t2[key] = 1
        else:
            self.b1.pop(key, None), self.b2.pop(key, None)
            self.t1[key] = 1

    def demote(self, key):
        (self.t1 if key in self.t1 else self.t2).move_to_end(key, last=False)

    def remove(self, key):
        (self.t1 if key in self.t1 else self.t2).pop(key)

    def evict(self):
        if self.t1 and (len(self.t1) > self.p or not self.t2):
            key = self.t1.popitem(last=False)[0]
            self.b1[key] = 1
        else:
            key = self.t2.popitem(last=False)[0]
            self.b2[key] = 1
        while len(self.b1) + len(self.b2) > self.cap:
            (self.b1 if len(self.b1) > len(self.b2) else self.b2).popitem(last=False)
        return key


class Score(LRU):
    """Evicts the cached expert with the lowest selection score (see the module's docstring),
    except the layer being served and read-aheads whose layer has not come yet (`pending`, set
    by the replay). Demotion does nothing: a read-ahead keeps the score its selections gave it."""

    def __init__(self, cap, half=HALF):
        self.cap, self.grow = cap, 2 ** (1 / half)
        self.s, self.cached, self.n = np.zeros(LAYERS * EXPERTS), np.zeros(LAYERS * EXPERTS, bool), 0
        self.serving, self.pending = (), {}

    def observe(self, t, l, act):
        self.serving = [l * EXPERTS + e for e in act]
        self.s[self.serving] += self.grow ** t

    def __contains__(self, key):
        return self.cached[key]

    def __len__(self):
        return self.n

    def use(self, key):
        pass

    def add(self, key, demand):
        self.cached[key] = True
        self.n += 1

    def demote(self, key):
        pass

    def remove(self, key):
        self.cached[key] = False
        self.n -= 1

    def evict(self):
        v = np.where(self.cached, self.s, np.inf)
        v[self.serving] = np.inf
        v[list(self.pending)] = np.inf
        key = int(v.argmin())
        self.remove(key)
        return key


def timed(trace, cap, sources, refine=True, demote=True, cache=Score):
    """sources: (name, ahead, k, admit); ahead 0 reads at START, the others at the routing, for
    the layer `ahead` layers on, which may belong to a later token.
    Returns ms per token, demand misses, read-aheads and used read-aheads per token."""
    c, pending, ready, queue = cache(cap), {}, {}, []  # ready < 0: queued at -ready, not started
    c.pending = pending
    clock = ssd = 0.0
    total = miss = reads = used = 0.0

    def drain(until):
        nonlocal ssd
        while queue and ssd < until:
            key = queue.pop(0)
            ssd = max(ssd, -ready[key]) + READ
            ready[key] = ssd

    def read_ahead(t, l, rec, at, early, counted):
        nonlocal reads
        for name, ahead, k, admit in sources:
            tt, layer = divmod(t * LAYERS + l + ahead, LAYERS)
            if (ahead == 0) != early or tt >= len(trace) or (tt > t and not name.startswith("oracle")):
                continue
            if refine:
                top = {layer * EXPERTS + e for e in rec[name][:KEEP]}
                for key, target in list(pending.items()):
                    if target == (tt, layer) and ready.get(key, 0) < 0 and key not in top:
                        queue.remove(key)
                        del pending[key], ready[key]
                        c.remove(key)
                        reads -= counted
            left = admit
            for e in rec[name][:k]:
                key = layer * EXPERTS + e
                if left == 0:
                    break
                if key in c:
                    continue
                c.add(key, False)
                pending[key] = (tt, layer)
                ready[key] = -at
                queue.append(key)
                reads += counted
                left -= 1

    def evict():
        while len(c) > cap:
            key = c.evict()
            pending.pop(key, None)
            if ready.pop(key, 0) < 0:
                queue.remove(key)

    for t, tok in enumerate(trace):
        counted = t >= WARM
        t0 = clock
        for l, rec in enumerate(tok):
            if hasattr(c, "observe"):
                c.observe(t, l, rec["act"])
            read_ahead(t, l, rec, clock + START, True, counted)
            evict()
            routed = clock + FRONT
            drain(routed)
            need = routed
            for e in rec["act"]:
                key = l * EXPERTS + e
                if key in c:
                    c.use(key)
                    if pending.pop(key, None) is not None:
                        used += counted
                    r = ready.get(key, 0)
                    if r < 0:  # queued, not started: read now, on demand
                        queue.remove(key)
                        ssd = max(ssd, routed) + READ
                        ready[key] = r = ssd
                    need = max(need, r)
                else:
                    miss += counted
                    c.add(key, True)
                    ssd = max(ssd, routed) + READ
                    ready[key] = ssd
                    need = max(need, ssd)
            for key, target in list(pending.items()):
                if target == (t, l):
                    del pending[key]
                    if demote and key in c:
                        c.demote(key)
            read_ahead(t, l, rec, routed, False, counted)
            evict()
            clock = need + EXPERT
        if counted:
            total += clock - t0
    n = len(trace) - WARM
    return total / n * 1e3, miss / n, reads / n, used / n


def read_trace(path):
    words = np.fromfile(path, np.int32)
    words = words[:len(words) // (LAYERS * RECORD) * LAYERS * RECORD].reshape(-1, LAYERS, RECORD)
    ids = words[:, :, :TOPK + len(NAMES) * PREDICT].tolist()
    trace = []
    for t in range(len(ids)):
        tok = []
        for l in range(LAYERS):
            o = ids[t][l]
            rec = {"act": o[:TOPK]}
            for i, name in enumerate(NAMES):
                rec[name] = o[TOPK + i * PREDICT:TOPK + (i + 1) * PREDICT]
            tok.append(rec)
        trace.append(tok)
    for t, tok in enumerate(trace):  # oracles: the actual experts a layers on
        for l, rec in enumerate(tok):
            for a in LEADS:
                tt, ll = divmod(t * LAYERS + l + a, LAYERS)
                rec[f"oracle{a}"] = trace[tt][ll]["act"] if tt < len(trace) else []
    return trace


POLICIES = {  # name: sources, refinement and demotion
    "none": ([], False, False),
    "myds4": ([("pc", 0, 3, 3), ("p1", 1, 3, 3), ("p2", 2, 3, 3)], True, True),
    **{f"oracle {a} layers ahead": ([(f"oracle{a}", a, 6, 6)], False, True) for a in LEADS},
}


def main(path, *args):
    global FRONT, EXPERT
    args = list(args)
    for name in ("--front", "--expert"):
        if name in args:
            i = args.index(name)
            globals()[name[2:].upper()] = float(args[i + 1]) * 1e-3
            del args[i:i + 2]
    slots = args
    trace = read_trace(path)
    n = len(trace) - WARM
    print(f"{len(trace)} tokens traced, {n} counted")
    for name, ahead in (("p1", 1), ("p2", 2), ("pc", 0)):
        print(f"recall of {name} ({ahead} ahead): top 3 {recall(trace, name, ahead, 3):.1%}, "
              f"top 6 {recall(trace, name, ahead, 6):.1%}, top 16 {recall(trace, name, ahead, 16):.1%}")
    reqs, warm = demand_requests(trace), WARM * LAYERS * TOPK
    for s in map(int, slots):
        line = [f"{name} {f(reqs, s, warm) / n:.1f}" for name, f in
                (("LRU", lru), ("LFU", lfu), ("LRU-2", lru2), ("ARC", arc), ("SLRU", slru), ("optimal", belady))]
        line.append(f"scores {scores(trace, s, WARM) / n:.1f}")
        print(f"{s:5d} slots ({s * 18.87e6 / 2**30:.0f} GiB), demand-only misses/token: " + ", ".join(line))
        for name, (sources, refine, demote) in POLICIES.items():
            ms, m, rd, u = timed(trace, s, sources, refine, demote)
            print(f"  {name:24s} {ms:5.1f} ms/token ({1e3 / ms:4.1f} tokens/s); demand {m:5.1f}, "
                  f"read ahead {rd:5.1f} ({u:4.1f} used)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
