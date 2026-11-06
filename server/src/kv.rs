//! The KV prefix cache (docs/server.md): which attention states and snapshots
//! the server keeps between requests, and how a prompt reuses them.
//!
//! Rows stay in the engine's states that computed them. The cache holds plain
//! data about them: each state's token ids (the positions whose rows it
//! holds), its snapshots' positions and sizes, its bytes, whether a request
//! decodes in it, and when it was last used. [`Cache::plan`] matches a prompt
//! against every state, busy or idle, and picks the cheapest way to resume at
//! the longest shared prefix `k` (every way exact without --ced):
//!
//! - **In place**: an idle state that the prompt extends, whole or up to the
//!   engine's rewind margin before its end, continues without a copy.
//! - **Snapshot**: a new state copies a source's rows before a snapshot's
//!   position `b <= k`, restores the snapshot and prefills from `b`; the
//!   prefill captures the rolling rows at `k` when `b < k` (a branch snapshot).
//! - **Replay**: a new state copies a source's rows before `k` and prefills
//!   from `k` with the engine's replay (exact, or bounded to a ring under
//!   --ced), which rebuilds the rolling rows at `k` (and gives them as a
//!   branch snapshot).
//! - **Cold**: nothing is shared; a new state prefills from 0.
//!
//! The worker carries out the plan with engine calls and reports back what it
//! made ([`Cache::add`], [`Cache::add_snap`], [`Cache::push`]). The cache owns no
//! engine handles, so its tests need no model.

use std::collections::BTreeMap;

/// A state's key; the worker maps it to its engine state.
pub type Id = u64;

/// Where a snapshot was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// At a prompt's end, with the logits its next token is sampled from.
    PromptEnd,
    /// At a generation's end.
    GenerationEnd,
    /// At a fork's branch point, captured during its prefill (a replay's
    /// resume point, or where the prompt parts from every state).
    Branch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snap {
    pub pos: usize,
    pub kind: Kind,
    pub bytes: u64,
}

impl Snap {
    pub fn logits(&self) -> bool {
        self.kind == Kind::PromptEnd
    }
}

/// One attention state as the cache sees it.
#[derive(Debug)]
pub struct Entry {
    /// The tokens whose rows the state holds, from position 0.
    pub tokens: Vec<i32>,
    /// A request decodes in it: its rows before its end are copyable, but it
    /// is never continued in place.
    pub busy: bool,
    /// The order of last use, for least-recently-used eviction.
    pub used: u64,
    /// The bytes its rows take (myds4_state_bytes), the snapshots apart.
    pub bytes: u64,
    pub snaps: Vec<Snap>,
    /// Its number in the log and the live view.
    pub number: u64,
    /// The positions requests reused from it, the last 32 of them.
    pub hits: Vec<usize>,
}

/// How a prompt resumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// A new state, prefilled from position 0.
    Cold,
    /// The idle state itself continues at `from`: with `from` equal to the
    /// prompt's length, from its prompt-end snapshot there (no prefill).
    InPlace { state: Id, from: usize },
    /// A new state: the source's rows before `at`, its snapshot at `at`, then
    /// a prefill from `at` (none when `at` is the prompt's length: the
    /// snapshot's logits are the prompt's). `branch` is where the prompt
    /// parts from every state (the longest prefix it shares with one), at or
    /// after `at`: the prefill captures the rolling rows there when it lies
    /// between `at` and the prompt's end, so a later fork there resumes at once.
    Snapshot { source: Id, at: usize, branch: usize },
    /// A new state: the source's rows before `at`, then a replay at `at` (exact,
    /// or bounded with --ced), which captures the rolling rows there.
    Replay { source: Id, at: usize },
}

impl Plan {
    /// The prompt tokens it reuses: where its prefill starts.
    pub fn reused(&self) -> usize {
        match *self {
            Plan::Cold => 0,
            Plan::InPlace { from, .. } => from,
            Plan::Snapshot { at, .. } | Plan::Replay { at, .. } => at,
        }
    }

    /// The state whose rows it reads, if any.
    pub fn source(&self) -> Option<Id> {
        match *self {
            Plan::Cold => None,
            Plan::InPlace { state, .. } => Some(state),
            Plan::Snapshot { source, .. } | Plan::Replay { source, .. } => Some(source),
        }
    }

    /// The log's name for it: new, kept (the state itself continues), saved (the
    /// same prompt again, from its snapshot) or forked (a copy of another state).
    pub fn kind(&self, n: usize) -> &'static str {
        match *self {
            Plan::Cold => "new",
            Plan::InPlace { from, .. } if from == n => "saved",
            Plan::InPlace { .. } => "kept",
            Plan::Snapshot { at, .. } if at == n => "saved",
            Plan::Snapshot { .. } | Plan::Replay { .. } => "forked",
        }
    }

    /// Where the fork's prefill captures its branch point, so that a later
    /// fork there resumes at once: a replay's resume point, or where the
    /// prompt parts from every state when that lies after the snapshot it
    /// resumes from and before the prompt's end.
    pub fn capture(&self, n: usize) -> Option<usize> {
        match *self {
            Plan::Replay { at, .. } => Some(at),
            Plan::Snapshot { at, branch, .. } if branch > at && branch < n => Some(branch),
            _ => None,
        }
    }

    /// The snapshot bytes a request on this plan takes before it decodes:
    /// its prompt end's (`with_logits` each) when it prefills (resuming at
    /// the prompt's end, it uses the snapshot there), and its branch point's
    /// (`snap`).
    pub fn snapshot_need(&self, n: usize, snap: u64, with_logits: u64) -> u64 {
        let prompt_end = if self.reused() < n { with_logits } else { 0 };
        prompt_end + if self.capture(n).is_some() { snap } else { 0 }
    }
}

/// The engine's limits and costs that plans depend on.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// The positions a state can continue before its end (myds4_rewind).
    pub rewind: usize,
    /// Whether a replay is available.
    pub replay: bool,
    /// What a replay at k costs beyond its own new tokens, in prefilled
    /// tokens: the rows before k each layer computes again, on average
    /// (myds4_replay_rows: exact, at most k, about 2,610 for long prefixes;
    /// bounded with --ced, the ring's 133).
    pub replay_cost: fn(usize) -> usize,
}

/// The positions two token sequences share from the start.
pub fn common(a: &[i32], b: &[i32]) -> usize {
    let n = a.len().min(b.len());
    let mut k = 0;
    while k + 1024 <= n && a[k..k + 1024] == b[k..k + 1024] {
        k += 1024;
    }
    while k < n && a[k] == b[k] {
        k += 1;
    }
    k
}

/// What [`Cache::fit`] frees: snapshots by state and position, then whole states.
#[derive(Debug, Default, PartialEq)]
pub struct Eviction {
    pub snaps: Vec<(Id, usize)>,
    pub states: Vec<Id>,
}

/// What the cache gave up, for the worker to free: snapshots, and whole
/// states with their snapshots.
pub type Freed = (Vec<(Id, Snap)>, Vec<(Id, Vec<Snap>)>);

#[derive(Debug)]
pub struct Cache {
    pub entries: BTreeMap<Id, Entry>,
    /// --kv-cache: what states and snapshots may take, in bytes.
    pub limit: u64,
    /// At most this many states.
    pub max_states: usize,
    clock: u64,
    next: Id,
    numbers: u64,
}

impl Cache {
    pub fn new(limit: u64, max_states: usize) -> Self {
        Cache { entries: BTreeMap::new(), limit, max_states, clock: 0, next: 1, numbers: 0 }
    }

    /// The bytes all states and snapshots take.
    pub fn bytes(&self) -> u64 {
        self.entries.values().map(|e| e.bytes + e.snaps.iter().map(|s| s.bytes).sum::<u64>()).sum()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The cheapest way to resume the prompt `p` (exact, but for a replay
    /// under --ced).
    pub fn plan(&self, p: &[i32], limits: Limits) -> Plan {
        let n = p.len();
        let mut best = (n, 3usize, Plan::Cold); // cost, then preference: in place, snapshot, replay, cold
        let mut branch = 0; // where the prompt parts from every state
        let mut consider = |cost: usize, rank: usize, plan: Plan| {
            if (cost, rank) < (best.0, best.1) {
                best = (cost, rank, plan);
            }
        };
        for (&id, e) in &self.entries {
            let k = common(p, &e.tokens);
            branch = branch.max(k.min(n));
            if k == 0 {
                continue;
            }
            // The whole prompt matches: without its logits there, its last token runs again.
            let k = if k >= n && !e.snaps.iter().any(|s| s.pos == n && s.logits()) { n - 1 } else { k.min(n) };
            if k == 0 {
                continue;
            }
            if !e.busy && k + limits.rewind >= e.tokens.len() {
                consider(n - k, 0, Plan::InPlace { state: id, from: k });
            }
            for s in &e.snaps {
                if s.pos <= k && (s.pos < n || s.logits()) {
                    consider(n - s.pos, 1, Plan::Snapshot { source: id, at: s.pos, branch: 0 });
                }
            }
            if limits.replay && k < n {
                consider(n - k + (limits.replay_cost)(k), 2, Plan::Replay { source: id, at: k });
            }
        }
        match best.2 {
            Plan::Snapshot { source, at, .. } => Plan::Snapshot { source, at, branch: branch.max(at) },
            plan => plan,
        }
    }

    /// A new state's entry, empty and busy; its id.
    pub fn add(&mut self, bytes: u64) -> Id {
        let id = self.next;
        self.next += 1;
        self.numbers += 1;
        let used = self.tick();
        self.entries.insert(
            id,
            Entry { tokens: Vec::new(), busy: true, used, bytes, snaps: Vec::new(), number: self.numbers, hits: Vec::new() },
        );
        id
    }

    /// A request starts in `id` (just added, or continued in place) at `from`:
    /// it is busy and holds the prompt (its rows before `from` copied or its
    /// own, the others prefilled). Its rows after `from` are overwritten, so
    /// the snapshots past `from` are dropped: returned for the worker to free.
    pub fn start(&mut self, id: Id, p: &[i32], from: usize, bytes: u64) -> Vec<Snap> {
        let used = self.tick();
        let e = self.entries.get_mut(&id).expect("a live state");
        e.busy = true;
        e.used = used;
        e.bytes = bytes;
        e.tokens.clear();
        e.tokens.extend_from_slice(p);
        let (keep, dropped): (Vec<Snap>, Vec<Snap>) = e.snaps.drain(..).partition(|s| s.pos <= from);
        e.snaps = keep;
        dropped
    }

    /// The source of a fork: it is used now, and requests reused it up to `at`.
    pub fn touch(&mut self, id: Id, at: usize) {
        let used = self.tick();
        if let Some(e) = self.entries.get_mut(&id) {
            e.used = used;
            if e.hits.len() == 32 {
                e.hits.remove(0);
            }
            e.hits.push(at);
        }
    }

    pub fn add_snap(&mut self, id: Id, snap: Snap) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.snaps.retain(|s| s.pos != snap.pos);
            e.snaps.push(snap);
            e.snaps.sort_by_key(|s| s.pos);
        }
    }

    pub fn snap(&self, id: Id, pos: usize) -> Option<Snap> {
        self.entries.get(&id)?.snaps.iter().find(|s| s.pos == pos).copied()
    }

    /// A decoded token's rows are in the state.
    pub fn push(&mut self, id: Id, token: i32) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.tokens.push(token);
        }
    }

    /// Its request ended: the state is idle, used now, and takes `bytes`.
    pub fn finish(&mut self, id: Id, bytes: u64) {
        let used = self.tick();
        if let Some(e) = self.entries.get_mut(&id) {
            e.busy = false;
            e.used = used;
            e.bytes = bytes;
        }
    }

    /// How far the states and snapshots exceed the limit, in bytes.
    pub fn over(&self) -> u64 {
        self.bytes().saturating_sub(self.limit)
    }

    /// Before a pass that makes the busy state `id` take `bytes` in all (the
    /// engine grows a state when its positions need more room): the state is
    /// charged its new size now, and the cache gives up first what keeps it
    /// within its limit, as [`Cache::fit`] picks it. When that is not enough,
    /// everything else that can go goes (busy states stay), and the state
    /// grows past the limit; [`Cache::over`] then tells by how much.
    pub fn grow(&mut self, id: Id, bytes: u64) -> Freed {
        let Some(have) = self.entries.get(&id).map(|e| e.bytes) else {
            return Freed::default();
        };
        if bytes <= have {
            return Freed::default();
        }
        let (ev, _) = self.eviction(bytes - have, false, &[id]);
        let freed = self.evict(&ev);
        if let Some(e) = self.entries.get_mut(&id) {
            e.bytes = bytes;
        }
        freed
    }

    /// The request in `id` ended: the state is idle, used now, and takes
    /// `bytes`; `end` is its generation-end snapshot, if it has none there
    /// yet. The snapshot is kept only if it fits within the limit once what
    /// [`Cache::fit`] picks goes; otherwise it is not taken, and the cache
    /// only gets back within its limit if a state grew past it. Whether the
    /// worker takes the snapshot, and what to free.
    pub fn end(&mut self, id: Id, bytes: u64, end: Option<Snap>) -> (bool, Freed) {
        self.finish(id, bytes);
        if let Some(ev) = self.fit(end.map_or(0, |s| s.bytes), false, &[id]) {
            let freed = self.evict(&ev);
            if let Some(s) = end {
                self.add_snap(id, s);
            }
            return (end.is_some(), freed);
        }
        let freed = self.fit(0, false, &[id]).map(|ev| self.evict(&ev)).unwrap_or_default();
        (false, freed)
    }

    /// Removes a state's entry; its snapshots, for the worker to free.
    pub fn remove(&mut self, id: Id) -> Vec<Snap> {
        self.entries.remove(&id).map(|e| e.snaps).unwrap_or_default()
    }

    /// A state's spare snapshots: all but its last prompt end and its last
    /// branch point, oldest first.
    fn spare(e: &Entry) -> Vec<Snap> {
        let last = |k: Kind| e.snaps.iter().filter(|s| s.kind == k).map(|s| s.pos).max();
        let (prompt, branch) = (last(Kind::PromptEnd), last(Kind::Branch));
        e.snaps
            .iter()
            .filter(|s| !(s.kind == Kind::PromptEnd && Some(s.pos) == prompt) && !(s.kind == Kind::Branch && Some(s.pos) == branch))
            .copied()
            .collect()
    }

    /// Makes room for `need` more bytes and one more state (if `new_state`),
    /// keeping the states in `keep` whole and every busy state: spare snapshots
    /// go first (least recently used states first), then whole idle states.
    /// None, with nothing evicted, if it cannot fit; otherwise what to free.
    pub fn fit(&self, need: u64, new_state: bool, keep: &[Id]) -> Option<Eviction> {
        let (ev, fits) = self.eviction(need, new_state, keep);
        fits.then_some(ev)
    }

    /// What [`Cache::fit`] frees, in its order, until the room is there or
    /// nothing else can go; and whether the room is there.
    fn eviction(&self, need: u64, new_state: bool, keep: &[Id]) -> (Eviction, bool) {
        let mut by_use: Vec<(&Id, &Entry)> = self.entries.iter().collect();
        by_use.sort_by_key(|(_, e)| e.used);
        let mut total = self.bytes() + need;
        let mut states = self.entries.len() + usize::from(new_state);
        let mut out = Eviction::default();
        for &(&id, e) in &by_use {
            if total <= self.limit {
                break;
            }
            if keep.contains(&id) {
                continue;
            }
            for s in Self::spare(e) {
                if total <= self.limit {
                    break;
                }
                total -= s.bytes;
                out.snaps.push((id, s.pos));
            }
        }
        for &(&id, e) in &by_use {
            if total <= self.limit && states <= self.max_states {
                break;
            }
            if keep.contains(&id) || e.busy {
                continue;
            }
            let held: u64 = e.snaps.iter().filter(|s| !out.snaps.contains(&(id, s.pos))).map(|s| s.bytes).sum();
            total -= e.bytes + held;
            states -= 1;
            out.snaps.retain(|&(sid, _)| sid != id);
            out.states.push(id);
        }
        let fits = total <= self.limit && states <= self.max_states;
        (out, fits)
    }

    /// Applies an eviction from [`Cache::fit`]: the snapshots and states to free.
    pub fn evict(&mut self, ev: &Eviction) -> (Vec<(Id, Snap)>, Vec<(Id, Vec<Snap>)>) {
        let mut snaps = Vec::new();
        for &(id, pos) in &ev.snaps {
            if let Some(e) = self.entries.get_mut(&id)
                && let Some(i) = e.snaps.iter().position(|s| s.pos == pos)
            {
                snaps.push((id, e.snaps.remove(i)));
            }
        }
        let states = ev.states.iter().map(|&id| (id, self.remove(id))).collect();
        (snaps, states)
    }
}

/// One run of positions that a set of states share, for the live view's tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub lines: String,
    pub start: usize,
    pub end: usize,
    pub states: usize,
    pub hits: usize,
    /// The state ending here (its number), 0 for a shared prefix.
    pub number: u64,
    pub busy: bool,
    /// Its last prompt-end snapshot's position, -1 for none.
    pub saved: i64,
    pub used: u64,
    pub bytes: u64,
}

impl Cache {
    /// The prefix tree of the `most` most recently used states, depth first,
    /// at most `cap` nodes.
    pub fn tree(&self, most: usize, cap: usize) -> Vec<Node> {
        let mut set: Vec<&Entry> = self.entries.values().collect();
        set.sort_by_key(|e| std::cmp::Reverse(e.used));
        set.truncate(most);
        let mut out = Vec::new();
        if !set.is_empty() {
            Self::branch(&set, 0, "", true, true, cap, &mut out);
        }
        out
    }

    fn branch(set: &[&Entry], from: usize, indent: &str, last: bool, root: bool, cap: usize, out: &mut Vec<Node>) {
        if out.len() >= cap || indent.chars().count() > 48 {
            return;
        }
        let mut end = set[0].tokens.len();
        for e in &set[1..] {
            end = end.min(from + common(&set[0].tokens[from.min(set[0].tokens.len())..], &e.tokens[from.min(e.tokens.len())..]));
        }
        let mut next = indent.to_string();
        if end > from || set.len() == 1 {
            let mut x = Node {
                lines: format!("{indent}{}", if root { "" } else if last { "└─ " } else { "├─ " }),
                start: from,
                end,
                states: set.len(),
                hits: set.iter().map(|e| e.hits.iter().filter(|&&h| h > from).count()).sum(),
                number: 0,
                busy: false,
                saved: -1,
                used: 0,
                bytes: 0,
            };
            if let Some(e) = set.iter().find(|e| e.tokens.len() == end) {
                x.number = e.number;
                x.busy = e.busy;
                x.saved = e.snaps.iter().filter(|s| s.kind == Kind::PromptEnd).map(|s| s.pos as i64).max().unwrap_or(-1);
                x.used = e.used;
                x.bytes = e.bytes;
            }
            out.push(x);
            next = format!("{indent}{}", if root { "" } else if last { "   " } else { "│  " });
        }
        let rest: Vec<&Entry> = set.iter().copied().filter(|e| e.tokens.len() > end).collect();
        let mut taken = vec![false; rest.len()];
        for i in 0..rest.len() {
            if taken[i] {
                continue;
            }
            let mut group = Vec::new();
            for k in i..rest.len() {
                if !taken[k] && rest[k].tokens[end] == rest[i].tokens[end] {
                    group.push(rest[k]);
                    taken[k] = true;
                }
            }
            let left = taken.iter().skip(i + 1).any(|t| !t);
            Self::branch(&group, end, &next, !left, false, cap, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// myds4_replay_rows of the model: 40 layers, layer 39 back a ring of 133
    /// rows, each layer below a window of 127 more.
    fn replay_rows(k: usize) -> usize {
        (0..40).map(|j| k.min(133 + 127 * j)).sum::<usize>().div_ceil(40)
    }

    const LIMITS: Limits = Limits { rewind: 5, replay: true, replay_cost: replay_rows };
    const SNAP: u64 = 6 << 20;

    fn seq(range: std::ops::Range<i32>) -> Vec<i32> {
        range.collect()
    }

    /// A state holding `tokens`, idle, with snapshots at `snaps` (prompt ends).
    fn state(c: &mut Cache, tokens: &[i32], snaps: &[usize]) -> Id {
        let id = c.add(1 << 20);
        c.start(id, tokens, 0, 1 << 20);
        for &pos in snaps {
            c.add_snap(id, Snap { pos, kind: Kind::PromptEnd, bytes: SNAP });
        }
        c.finish(id, 1 << 20);
        id
    }

    #[test]
    fn a_prompt_that_extends_an_idle_state_continues_it_in_place() {
        let mut c = Cache::new(u64::MAX, 256);
        let a = state(&mut c, &seq(0..1000), &[1000]);
        let p = seq(0..1200);
        assert_eq!(c.plan(&p, LIMITS), Plan::InPlace { state: a, from: 1000 });
        // within the rewind margin before its end: still in place
        let mut q = seq(0..997);
        q.extend(5000..5100);
        assert_eq!(c.plan(&q, LIMITS), Plan::InPlace { state: a, from: 997 });
    }

    #[test]
    fn the_longest_match_wins_whichever_state_holds_it_busy_or_not() {
        let mut c = Cache::new(u64::MAX, 256);
        let _short = state(&mut c, &seq(0..3000), &[3000]);
        let long = state(&mut c, &seq(0..20000), &[10000, 20000]);
        c.entries.get_mut(&long).unwrap().busy = true;
        let mut p = seq(0..15000);
        p.extend(90000..90010);
        // a busy state is a source like any other, but never continued in place
        assert_eq!(c.plan(&p, LIMITS), Plan::Replay { source: long, at: 15000 });
    }

    #[test]
    fn a_nearby_snapshot_beats_a_replay_and_a_far_one_loses_to_it() {
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..20000), &[9000, 19000]);
        let mut near = seq(0..19500);
        near.push(-1);
        // snapshot at 19000: 501 tokens; replay at 19500: 1 + 2600
        assert_eq!(c.plan(&near, LIMITS), Plan::Snapshot { source: s, at: 19000, branch: 19500 });
        let mut far = seq(0..15000);
        far.push(-1);
        // snapshot at 9000: 6001 tokens; replay at 15000: 1 + 2600
        assert_eq!(c.plan(&far, LIMITS), Plan::Replay { source: s, at: 15000 });
        // without the replay (--ced), the snapshot
        assert_eq!(c.plan(&far, Limits { replay: false, ..LIMITS }), Plan::Snapshot { source: s, at: 9000, branch: 15000 });
    }

    #[test]
    fn the_same_prompt_again_resumes_from_its_prompt_end_without_a_prefill() {
        let mut c = Cache::new(u64::MAX, 256);
        let p = seq(0..4000);
        let s = state(&mut c, &p, &[4000]);
        // its reply, decoded past the rewind margin
        for t in 0..300 {
            c.push(s, 100000 + t);
        }
        let plan = c.plan(&p, LIMITS);
        assert_eq!(plan, Plan::Snapshot { source: s, at: 4000, branch: 4000 });
        assert_eq!(plan.reused(), 4000);
        assert_eq!(plan.kind(p.len()), "saved");
        // without logits there, its last token runs again
        let mut d = Cache::new(u64::MAX, 256);
        let t = state(&mut d, &p, &[]);
        assert_eq!(d.plan(&p, LIMITS), Plan::InPlace { state: t, from: 3999 });
    }

    #[test]
    fn a_retry_after_a_short_reply_and_a_changed_last_token_stay_in_place() {
        let mut c = Cache::new(u64::MAX, 256);
        let p = seq(0..4000);
        let s = state(&mut c, &p, &[4000]);
        for t in 0..3 {
            c.push(s, 100000 + t); // a reply within the rewind margin
        }
        assert_eq!(c.plan(&p, LIMITS), Plan::InPlace { state: s, from: 4000 });
        assert_eq!(c.plan(&p, LIMITS).kind(p.len()), "saved");
        let mut changed = p.clone();
        *changed.last_mut().unwrap() = -7;
        assert_eq!(c.plan(&changed, LIMITS), Plan::InPlace { state: s, from: 3999 });
    }

    #[test]
    fn a_prompt_inside_a_longer_state_reuses_it_at_any_length() {
        assert_eq!(replay_rows(20000), 2610);
        assert_eq!((replay_rows(1999), replay_rows(100)), (1633, 100));
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..20000), &[4000]);
        // 12,000 of its tokens: the replay at 11,999 (1 + 2,610) beats the snapshot at 4,000
        assert_eq!(c.plan(&seq(0..12000), LIMITS), Plan::Replay { source: s, at: 11999 });
        // 2,000 of them: a replay recomputes 1,633 rows per layer, still less than a cold prefill
        assert_eq!(c.plan(&seq(0..2000), LIMITS), Plan::Replay { source: s, at: 1999 });
        // 100: a replay recomputes them all, as a cold prefill would, and reports them reused
        assert_eq!(c.plan(&seq(0..100), LIMITS), Plan::Replay { source: s, at: 99 });
    }

    #[test]
    fn a_fork_resumed_from_an_earlier_snapshot_carries_its_own_branch_point() {
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..6000), &[]);
        c.add_snap(s, Snap { pos: 4125, kind: Kind::Branch, bytes: SNAP }); // an earlier fork's
        let mut p = seq(0..4262);
        p.extend(-20..0);
        // 137 tokens from the snapshot cost less than a replay at 4,262 (~2,610 rows)
        assert_eq!(c.plan(&p, LIMITS), Plan::Snapshot { source: s, at: 4125, branch: 4262 });
    }

    #[test]
    fn under_ced_the_bounded_replay_wins_past_a_ring() {
        // myds4_replay_rows under --ced: the ring, 133 positions
        let ced = Limits { replay_cost: |k| k.min(133), ..LIMITS };
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..6000), &[]);
        c.add_snap(s, Snap { pos: 4125, kind: Kind::Branch, bytes: SNAP });
        let mut far = seq(0..4262);
        far.extend(-20..0);
        assert_eq!(c.plan(&far, ced), Plan::Replay { source: s, at: 4262 });
        let mut near = seq(0..4200);
        near.extend(-20..0);
        assert_eq!(c.plan(&near, ced), Plan::Snapshot { source: s, at: 4125, branch: 4200 });
    }

    #[test]
    fn nothing_shared_is_a_cold_prefill() {
        let mut c = Cache::new(u64::MAX, 256);
        state(&mut c, &seq(0..100), &[100]);
        assert_eq!(c.plan(&seq(500..700), LIMITS), Plan::Cold);
        assert_eq!(Cache::new(u64::MAX, 256).plan(&seq(0..10), LIMITS), Plan::Cold);
    }

    #[test]
    fn continuing_in_place_truncates_the_state_and_its_later_snapshots() {
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..1000), &[400, 998, 1000]);
        let mut p = seq(0..998);
        p.extend(7000..7050);
        assert_eq!(c.plan(&p, LIMITS), Plan::InPlace { state: s, from: 998 });
        let dropped = c.start(s, &p, 998, 1 << 20);
        assert_eq!(dropped.iter().map(|x| x.pos).collect::<Vec<_>>(), vec![1000]);
        let e = &c.entries[&s];
        assert_eq!(e.tokens, p);
        assert_eq!(e.snaps.iter().map(|x| x.pos).collect::<Vec<_>>(), vec![400, 998]);
        assert!(e.busy);
        // a later prompt on the old tail no longer matches it
        assert_eq!(common(&seq(0..1000), &e.tokens), 998);
    }

    #[test]
    fn a_replay_leaves_its_branch_point_as_a_snapshot_of_the_new_state() {
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..8000), &[8000]);
        let mut p = seq(0..5000);
        p.extend(-50..0);
        let Plan::Replay { source, at } = c.plan(&p, LIMITS) else { panic!("a replay") };
        assert_eq!((source, at), (s, 5000));
        let new = c.add(1 << 20);
        c.start(new, &p, at, 1 << 20);
        c.add_snap(new, Snap { pos: at, kind: Kind::Branch, bytes: SNAP });
        c.touch(source, at);
        c.finish(new, 1 << 20);
        // the next fork at the same point resumes from that snapshot
        let mut q = seq(0..5000);
        q.extend(-90..-60);
        assert_eq!(c.plan(&q, LIMITS), Plan::Snapshot { source: new, at: 5000, branch: 5000 });
    }

    #[test]
    fn a_request_that_cannot_fit_is_refused_before_anything_is_evicted() {
        let mut c = Cache::new(10 << 20, 256);
        let a = state(&mut c, &seq(0..10), &[]);
        let b = state(&mut c, &seq(20..30), &[]);
        assert_eq!(c.bytes(), 2 << 20);
        c.entries.get_mut(&a).unwrap().busy = true;
        // 10 MiB more: only b (1 MiB) can go, which is not enough
        assert_eq!(c.fit(10 << 20, true, &[]), None);
        assert_eq!(c.entries.len(), 2);
        // 9 MiB more fits once b goes
        assert_eq!(c.fit(9 << 20, true, &[]), Some(Eviction { snaps: vec![], states: vec![b] }));
    }

    #[test]
    fn spare_snapshots_go_before_states_and_old_states_before_new_ones() {
        let mut c = Cache::new(u64::MAX, 256);
        let old = state(&mut c, &seq(0..100), &[50, 100]); // 50 is spare: 100 is its last prompt end
        let new = state(&mut c, &seq(200..300), &[300 - 200]);
        c.entries.get_mut(&new).unwrap().snaps.push(Snap { pos: 40, kind: Kind::GenerationEnd, bytes: SNAP });
        let total = c.bytes();
        c.limit = total;
        // one snapshot's room: the oldest state's spare snapshot
        assert_eq!(c.fit(SNAP, false, &[]), Some(Eviction { snaps: vec![(old, 50)], states: vec![] }));
        // a source in use is kept: the other state's spare snapshot instead
        assert_eq!(c.fit(SNAP, false, &[old]), Some(Eviction { snaps: vec![(new, 40)], states: vec![] }));
        // more than the spare snapshots: the least recently used state goes whole
        let ev = c.fit(2 * SNAP + (1 << 20), false, &[]).unwrap();
        assert_eq!(ev.states, vec![old]);
        let (snaps, states) = c.evict(&ev);
        assert_eq!(states.len(), 1);
        assert!(snaps.iter().all(|(id, _)| *id == new));
        assert!(!c.entries.contains_key(&old));
    }

    #[test]
    fn a_saved_prompt_again_takes_no_new_snapshot_and_fits_a_full_budget() {
        let mut c = Cache::new(u64::MAX, 256);
        let p = seq(0..4000);
        let s = state(&mut c, &p, &[4000]);
        // the same prompt again resumes at its prompt end, without a prefill
        let plan = c.plan(&p, LIMITS);
        assert_eq!(plan.reused(), p.len());
        assert_eq!(plan.snapshot_need(p.len(), SNAP, SNAP + 1), 0);
        // so with the cache full it fits, and nothing is freed
        c.limit = c.bytes();
        assert_eq!(c.fit(plan.snapshot_need(p.len(), SNAP, SNAP + 1), false, &[s]), Some(Eviction::default()));
        // a fork prefills: its prompt end (with logits) and its branch point
        let q = [&p[..3000], &seq(9000..9100)[..]].concat();
        let fork = c.plan(&q, LIMITS);
        assert_eq!(fork.capture(q.len()), Some(3000));
        assert_eq!(fork.snapshot_need(q.len(), SNAP, SNAP + 1), 2 * SNAP + 1);
    }

    #[test]
    fn a_decoding_state_is_charged_its_growth_before_it_grows_and_others_go_first() {
        let mut c = Cache::new(u64::MAX, 256);
        let old = state(&mut c, &seq(0..10), &[10]);
        c.add_snap(old, Snap { pos: 5, kind: Kind::GenerationEnd, bytes: SNAP }); // spare
        let busy = state(&mut c, &seq(20..30), &[]);
        c.start(busy, &seq(20..30), 10, 1 << 20); // decoding
        c.limit = c.bytes();
        // one snapshot's growth: the spare snapshot goes, before the growth
        let (snaps, states) = c.grow(busy, (1 << 20) + SNAP);
        assert_eq!(snaps.iter().map(|&(id, s)| (id, s.pos)).collect::<Vec<_>>(), vec![(old, 5)]);
        assert!(states.is_empty());
        assert_eq!(c.entries[&busy].bytes, (1 << 20) + SNAP);
        assert_eq!(c.over(), 0);
        // more: the idle state goes whole
        let (_, states) = c.grow(busy, (3 << 20) + SNAP);
        assert_eq!(states.iter().map(|s| s.0).collect::<Vec<_>>(), vec![old]);
        assert_eq!(c.entries[&busy].bytes, (3 << 20) + SNAP);
        assert_eq!(c.over(), 0);
        // no growth: nothing happens
        let (snaps, states) = c.grow(busy, 1 << 20);
        assert!(snaps.is_empty() && states.is_empty());
        assert_eq!(c.entries[&busy].bytes, (3 << 20) + SNAP);
    }

    #[test]
    fn growth_that_cannot_fit_frees_every_idle_state_first_and_then_exceeds() {
        let mut c = Cache::new(u64::MAX, 256);
        let idle = state(&mut c, &seq(0..10), &[]);
        let a = state(&mut c, &seq(20..30), &[]);
        let b = state(&mut c, &seq(40..50), &[]);
        c.start(a, &seq(20..30), 10, 1 << 20);
        c.start(b, &seq(40..50), 10, 1 << 20);
        c.limit = c.bytes(); // 3 MiB
        // 3 MiB more: only the idle state (1 MiB) can go; the busy ones stay
        let (_, states) = c.grow(a, 4 << 20);
        assert_eq!(states.iter().map(|s| s.0).collect::<Vec<_>>(), vec![idle]);
        assert_eq!(c.entries[&a].bytes, 4 << 20);
        assert_eq!(c.over(), 2 << 20);
        assert!(c.entries.contains_key(&b));
    }

    #[test]
    fn a_generation_end_snapshot_is_kept_only_within_the_limit() {
        let end = Snap { pos: 10, kind: Kind::GenerationEnd, bytes: SNAP };
        // the cache full, nothing else to free: the snapshot is not taken
        let mut c = Cache::new(u64::MAX, 256);
        let s = state(&mut c, &seq(0..10), &[]);
        c.start(s, &seq(0..10), 10, 1 << 20);
        c.limit = c.bytes();
        let (take, _) = c.end(s, 1 << 20, Some(end));
        assert!(!take);
        assert!(c.snap(s, 10).is_none());
        assert_eq!(c.over(), 0);
        // an idle state to spare: it goes first and the snapshot is kept
        let mut d = Cache::new(u64::MAX, 256);
        let old = state(&mut d, &seq(100..110), &[]);
        let t = state(&mut d, &seq(0..10), &[]);
        d.start(t, &seq(0..10), 10, 1 << 20);
        d.limit = d.bytes() + SNAP - (1 << 20);
        let (take, (_, states)) = d.end(t, 1 << 20, Some(end));
        assert!(take);
        assert_eq!(states.iter().map(|s| s.0).collect::<Vec<_>>(), vec![old]);
        assert!(d.snap(t, 10).is_some());
        assert_eq!(d.over(), 0);
    }

    #[test]
    fn the_state_count_is_bounded_too() {
        let mut c = Cache::new(u64::MAX, 2);
        let a = state(&mut c, &seq(0..10), &[]);
        let _b = state(&mut c, &seq(20..30), &[]);
        assert_eq!(c.fit(0, true, &[]), Some(Eviction { snaps: vec![], states: vec![a] }));
        assert_eq!(c.fit(0, false, &[]), Some(Eviction::default()));
    }

    #[test]
    fn the_tree_shares_prefixes_and_parts_where_tokens_differ() {
        let mut c = Cache::new(u64::MAX, 256);
        let mut a = seq(0..100);
        a.extend(500..520);
        let mut b = seq(0..100);
        b.extend(600..650);
        let na = state(&mut c, &a, &[]);
        let nb = state(&mut c, &b, &[]);
        let t = c.tree(24, 40);
        assert_eq!(t.len(), 3);
        assert_eq!((t[0].start, t[0].end, t[0].states, t[0].number), (0, 100, 2, 0));
        let ends: Vec<(usize, u64)> = t[1..].iter().map(|x| (x.end, x.number)).collect();
        assert!(ends.contains(&(120, c.entries[&na].number)) && ends.contains(&(150, c.entries[&nb].number)));
        assert!(t[1].lines.starts_with("├─") && t[2].lines.starts_with("└─"));
    }
}
