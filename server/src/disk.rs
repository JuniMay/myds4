//! States saved on the SSD (kv.rs's [`Saved`]): their files in one folder
//! (kv_dir, docs/usage.md), and saving and reading them.
//!
//! A saved state `ID` is an index, `ID.kv`, and the files it names: its
//! parts, `ID-N.rows` (myds4_state_write), and its snapshots, `ID-V-POS.snap`
//! (written by its save V, at position POS). The index holds the model's
//! identity (myds4_state_identity), the tokens, each part's positions and
//! bytes, each snapshot's position, kind, stale rows and bytes, and a SHA-256
//! of the rest. Every file is written under a temporary name (`.tmp`), synced
//! and renamed, the index last: after a crash the folder holds the saved
//! state as it was or as it became, and files that no index names, which the
//! next start deletes. The store deletes no other files. Files are 0600, the
//! folder 0700. One server uses the folder at a time (a lock on its `lock`
//! file); another runs without saved states.

use crate::engine::{self, Model, RawSnapshot, RawState};
use crate::kv::{Cache, DiskId, Entry, Kind, Part, Save, Saved, SavedSnap, Snap};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::c_int;
use std::fs::{self, File};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

const MAGIC: &[u8; 8] = b"MYDS4KVI";
const VERSION: u32 = 1;
/// The bytes before an index's parts: magic, version, a reserved word, the
/// SHA-256 of what follows it, the identity, the save's version, the
/// replaced positions, the tokens', parts' and snapshots' counts.
const HEAD: usize = 112;
/// What a save leaves free on the volume.
const VOLUME_RESERVE: u64 = 10 << 30;
const GIB: f64 = (1u64 << 30) as f64;

/// One state's rows and snapshots as a save writes them and a load reads
/// them back: the engine's (engine::Model and one of its states), or a test's.
pub trait StateIo {
    /// The rows that positions `from..to` wrote, to `fd` (-1: their bytes
    /// alone); None when the write failed or a second stop ended it.
    fn rows(&self, from: usize, to: usize, fd: c_int) -> Option<u64>;
    /// Snapshot `x`, or (null) the state's end at `pos`, to `fd` (-1: its
    /// bytes alone).
    fn snapshot(&self, x: *const RawSnapshot, pos: usize, fd: c_int) -> Option<u64>;
    /// The model's identity (engine.h's myds4_state_identity).
    fn identity(&self) -> [u8; 32];
    /// The rows before `positions` from parts `fds` into the state; false
    /// when a part is short, damaged or of another shape.
    fn read_rows(&self, positions: usize, fds: &[c_int]) -> bool;
    /// A snapshot from `fd`; null when it is short, damaged or of another shape.
    fn read_snapshot(&self, fd: c_int) -> *mut RawSnapshot;
}

/// A state of an engine model.
pub struct Engine<'a> {
    pub model: &'a Model,
    pub state: *mut RawState,
}

impl StateIo for Engine<'_> {
    fn rows(&self, from: usize, to: usize, fd: c_int) -> Option<u64> {
        self.model.state_write(self.state, from, to, fd)
    }
    fn snapshot(&self, x: *const RawSnapshot, pos: usize, fd: c_int) -> Option<u64> {
        self.model.snapshot_write(x, self.state, pos, fd)
    }
    fn identity(&self) -> [u8; 32] {
        self.model.state_identity()
    }
    fn read_rows(&self, positions: usize, fds: &[c_int]) -> bool {
        self.model.state_read(self.state, positions, fds)
    }
    fn read_snapshot(&self, fd: c_int) -> *mut RawSnapshot {
        self.model.snapshot_read(fd)
    }
}

/// Why a saved state was not read: its files are damaged or missing (it
/// goes), or they could not be read now (it stays).
#[derive(Debug, PartialEq)]
pub enum LoadError {
    Damaged(String),
    Unavailable(String),
}

/// The folder, locked for this server, and what this run read and wrote.
pub struct Store {
    dir: PathBuf,
    _lock: File,
    next: DiskId,
    pub loads: u64,
    pub saves: u64,
    pub written: u64,
}

/// A saved state found at the start: its key, its model's identity, its
/// entry, and when it was last used.
pub struct Found {
    pub id: DiskId,
    pub identity: [u8; 32],
    pub saved: Saved,
    pub used: SystemTime,
}

/// The store's file names: an index, a part, a snapshot.
#[derive(Debug, PartialEq)]
enum Name {
    Index(DiskId),
    Part(DiskId),
    Snap(DiskId),
}

/// A file name of the store's (and whether it is a temporary one).
fn parse_name(name: &str) -> Option<(Name, bool)> {
    let (name, tmp) = match name.strip_suffix(".tmp") {
        Some(base) => (base, true),
        None => (name, false),
    };
    // a number as the store writes it: digits, no leading zero
    let number = |s: &str| {
        let plain = !s.is_empty() && s.len() <= 19 && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
        plain.then(|| s.parse::<u64>().ok()).flatten()
    };
    let (stem, kind) = name.rsplit_once('.')?;
    let fields: Vec<&str> = stem.split('-').collect();
    let id = number(fields[0])?;
    let parsed = match (kind, fields.len()) {
        ("kv", 1) => Name::Index(id),
        ("rows", 2) if number(fields[1]).is_some() => Name::Part(id),
        ("snap", 3) if number(fields[1]).is_some() && number(fields[2]).is_some() => Name::Snap(id),
        _ => return None,
    };
    Some((parsed, tmp))
}

fn part_name(id: DiskId, p: &Part) -> String {
    format!("{id}-{}.rows", p.file)
}

fn snap_name(id: DiskId, s: &SavedSnap) -> String {
    format!("{id}-{}-{}.snap", s.version, s.snap.pos)
}

fn kind_code(k: Kind) -> u8 {
    match k {
        Kind::PromptEnd => 0,
        Kind::GenerationEnd => 1,
        Kind::Branch => 2,
    }
}

/// A saved state's index.
pub fn encode(identity: &[u8; 32], s: &Saved) -> Vec<u8> {
    let mut b = Vec::with_capacity(HEAD + 24 * s.parts.len() + 32 * s.snaps.len() + 4 * s.tokens.len());
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&[0u8; 32]); // the SHA-256, below
    b.extend_from_slice(identity);
    b.extend_from_slice(&s.version.to_le_bytes());
    b.extend_from_slice(&(s.replaced as u64).to_le_bytes());
    b.extend_from_slice(&(s.tokens.len() as u64).to_le_bytes());
    b.extend_from_slice(&(s.parts.len() as u32).to_le_bytes());
    b.extend_from_slice(&(s.snaps.len() as u32).to_le_bytes());
    for p in &s.parts {
        b.extend_from_slice(&(p.from as u32).to_le_bytes());
        b.extend_from_slice(&(p.to as u32).to_le_bytes());
        b.extend_from_slice(&p.bytes.to_le_bytes());
        b.extend_from_slice(&p.file.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    for x in &s.snaps {
        b.extend_from_slice(&(x.snap.pos as u32).to_le_bytes());
        b.extend_from_slice(&[kind_code(x.snap.kind), 0, 0, 0]);
        b.extend_from_slice(&(x.snap.stale as u32).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&x.snap.bytes.to_le_bytes());
        b.extend_from_slice(&x.version.to_le_bytes());
    }
    for t in &s.tokens {
        b.extend_from_slice(&t.to_le_bytes());
    }
    let sum = Sha256::digest(&b[48..]);
    b[16..48].copy_from_slice(&sum);
    b
}

/// Whether an index is of another version of the store's: a server of
/// another version wrote it, and only that one reads it.
fn other_version(b: &[u8]) -> bool {
    b.len() >= 12 && &b[..8] == MAGIC && b[8..12] != VERSION.to_le_bytes()
}

/// An index's identity and entry (its model unknown yet); None when it is
/// damaged, of another version, or inconsistent.
pub fn decode(b: &[u8]) -> Option<([u8; 32], Saved)> {
    let u32_at = |at: usize| Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?));
    let u64_at = |at: usize| Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?));
    if b.len() < HEAD || &b[..8] != MAGIC || u32_at(8)? != VERSION || Sha256::digest(&b[48..])[..] != b[16..48] {
        return None;
    }
    let identity: [u8; 32] = b[48..80].try_into().ok()?;
    let (version, replaced, tokens) = (u64_at(80)?, u64_at(88)? as usize, u64_at(96)? as usize);
    let (parts, snaps) = (u32_at(104)? as usize, u32_at(108)? as usize);
    if b.len() != HEAD + 24 * parts + 32 * snaps + 4 * tokens {
        return None;
    }
    let mut at = HEAD;
    let mut saved = Saved {
        model: None,
        tokens: Vec::with_capacity(tokens),
        parts: Vec::with_capacity(parts),
        snaps: Vec::with_capacity(snaps),
        replaced,
        used: 0,
        version,
    };
    for _ in 0..parts {
        saved.parts.push(Part {
            from: u32_at(at)? as usize,
            to: u32_at(at + 4)? as usize,
            bytes: u64_at(at + 8)?,
            file: u32_at(at + 16)?,
        });
        at += 24;
    }
    for _ in 0..snaps {
        let kind = match b[at + 4] {
            0 => Kind::PromptEnd,
            1 => Kind::GenerationEnd,
            2 => Kind::Branch,
            _ => return None,
        };
        saved.snaps.push(SavedSnap {
            snap: Snap {
                pos: u32_at(at)? as usize,
                kind,
                bytes: u64_at(at + 16)?,
                stale: u32_at(at + 8)? as usize,
            },
            version: u64_at(at + 24)?,
        });
        at += 32;
    }
    for _ in 0..tokens {
        saved.tokens.push(u32_at(at)? as i32);
        at += 4;
    }
    // the parts continue each other up to the state's end; the snapshots lie
    // within it, one at a position
    let chained = saved.parts.first().is_some_and(|p| p.from == 0)
        && saved.parts.windows(2).all(|w| w[1].from <= w[0].to)
        && saved.parts.iter().all(|p| p.from <= p.to)
        && saved.parts.last().is_some_and(|p| p.to == tokens);
    let mut positions = HashSet::new();
    let placed = saved.snaps.iter().all(|s| s.snap.pos <= tokens && positions.insert(s.snap.pos));
    (chained && placed).then_some((identity, saved))
}

impl Store {
    /// The folder, made when missing and locked for this server, and the
    /// saved states in it, least recently used first. Damaged indexes,
    /// indexes whose files are missing or short, and files of the store's
    /// that no index names are deleted.
    pub fn open(dir: &Path) -> Result<(Store, Vec<Found>), String> {
        let shown = dir.display();
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .map_err(|e| format!("cannot make {shown}: {e}"))?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .mode(0o600)
            .open(dir.join("lock"))
            .map_err(|e| format!("cannot open {shown}/lock: {e}"))?;
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(format!("{shown} is in use by another server"));
        }
        let mut indexes = Vec::new();
        let mut files = Vec::new();
        let mut next = 1;
        for entry in fs::read_dir(dir).map_err(|e| format!("cannot read {shown}: {e}"))?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            match parse_name(&name) {
                Some((Name::Index(id), false)) => indexes.push(id),
                Some((Name::Index(id) | Name::Part(id) | Name::Snap(id), _)) => {
                    next = next.max(id + 1);
                    files.push((name, id));
                }
                None => {}
            }
        }
        let mut found = Vec::new();
        let mut named = HashSet::new();
        let mut others = HashSet::new(); // saved states of another version: theirs, kept
        for id in indexes {
            next = next.max(id + 1);
            let path = dir.join(format!("{id}.kv"));
            let used = fs::metadata(&path).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let bytes = fs::read(&path).unwrap_or_default();
            if other_version(&bytes) {
                engine::log(&format!("kv: {shown}/{id}.kv is another version's; kept, not read"));
                others.insert(id);
                continue;
            }
            let whole = |s: &Saved| {
                let size = |name: String| fs::metadata(dir.join(name)).map(|m| m.len()).ok();
                s.parts.iter().all(|p| size(part_name(id, p)) == Some(p.bytes))
                    && s.snaps.iter().all(|x| size(snap_name(id, x)) == Some(x.snap.bytes))
            };
            match decode(&bytes) {
                Some((identity, saved)) if whole(&saved) => {
                    named.extend(saved.parts.iter().map(|p| part_name(id, p)));
                    named.extend(saved.snaps.iter().map(|x| snap_name(id, x)));
                    found.push(Found { id, identity, saved, used });
                }
                _ => {
                    engine::log(&format!("kv: {shown}/{id}.kv is damaged or misses files; deleted"));
                    let _ = fs::remove_file(&path);
                }
            }
        }
        for (name, id) in files {
            if !named.contains(&name) && !others.contains(&id) {
                let _ = fs::remove_file(dir.join(&name));
            }
        }
        found.sort_by_key(|f| f.used);
        let store = Store {
            dir: dir.to_path_buf(),
            _lock: lock,
            next,
            loads: 0,
            saves: 0,
            written: 0,
        };
        Ok((store, found))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The bytes free on the folder's volume for this user.
    pub fn available(&self) -> u64 {
        let Ok(path) = std::ffi::CString::new(self.dir.as_os_str().as_encoded_bytes()) else {
            return 0;
        };
        let mut st: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(path.as_ptr(), &mut st) } != 0 {
            return 0;
        }
        st.f_bavail as u64 * st.f_bsize as u64
    }

    /// A new file under its temporary name.
    fn create(&self, name: &str) -> Result<(File, PathBuf, PathBuf), String> {
        let (path, tmp) = (self.dir.join(name), self.dir.join(format!("{name}.tmp")));
        let _ = fs::remove_file(&tmp);
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        Ok((file, tmp, path))
    }

    /// A written file, synced, under its own name.
    fn commit(file: File, tmp: &Path, path: &Path) -> Result<(), String> {
        file.sync_all().map_err(|e| format!("cannot sync {}: {e}", tmp.display()))?;
        fs::rename(tmp, path).map_err(|e| format!("cannot rename {}: {e}", tmp.display()))
    }

    /// Writes a file through `write` (its bytes, or None) under its
    /// temporary name and renames it; its bytes.
    fn write_file(&self, name: &str, write: impl FnOnce(c_int) -> Option<u64>) -> Result<u64, String> {
        let (file, tmp, path) = self.create(name)?;
        let Some(bytes) = write(file.as_raw_fd()) else {
            drop(file);
            let _ = fs::remove_file(&tmp);
            return Err(format!("cannot write {name} (or a second stop ended it)"));
        };
        Self::commit(file, &tmp, &path).inspect_err(|_| {
            let _ = fs::remove_file(&tmp);
        })?;
        Ok(bytes)
    }

    fn write_index(&self, id: DiskId, identity: &[u8; 32], s: &Saved) -> Result<(), String> {
        let (mut file, tmp, path) = self.create(&format!("{id}.kv"))?;
        let written = std::io::Write::write_all(&mut file, &encode(identity, s))
            .map_err(|e| format!("cannot write {}: {e}", tmp.display()))
            .and_then(|_| Self::commit(file, &tmp, &path));
        if written.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        // the rename synced before any file it replaces is deleted
        if let Ok(folder) = File::open(&self.dir) {
            unsafe { libc::fsync(folder.as_raw_fd()) };
        }
        written
    }

    /// Saved state `id`'s last use, for the next start: its index's
    /// modification time.
    fn set_used(&self, id: DiskId, when: SystemTime) {
        if let Ok(index) = fs::OpenOptions::new().write(true).open(self.dir.join(format!("{id}.kv"))) {
            let _ = index.set_modified(when);
        }
    }

    /// Deletes files of saved state `id`.
    fn remove_files(&self, id: DiskId, parts: &[Part], snaps: &[SavedSnap]) {
        for p in parts {
            let _ = fs::remove_file(self.dir.join(part_name(id, p)));
        }
        for x in snaps {
            let _ = fs::remove_file(self.dir.join(snap_name(id, x)));
        }
    }

    /// Deletes saved state `id`, its index first, and its entry.
    pub fn forget(&mut self, cache: &mut Cache, id: DiskId) {
        if let Some(d) = cache.saved.remove(&id) {
            let _ = fs::remove_file(self.dir.join(format!("{id}.kv")));
            self.remove_files(id, &d.parts, &d.snaps);
        }
    }

    /// Writes the evicted (or, at quit, idle) state `e`, whose rows `io`
    /// writes, with its snapshots `snapshot(pos)`, as the cache decides
    /// ([`Cache::save_decision`], `kept`: whether the states in the cache stay
    /// in memory), and records it in the cache. The saved state keeps the
    /// state's last use, `used` (also its index's modification time), so that
    /// room goes from the state used longest ago, also among those saved at
    /// quit. The log's line, if any.
    pub fn save(
        &mut self,
        cache: &mut Cache,
        io: &dyn StateIo,
        snapshot: &dyn Fn(usize) -> *mut RawSnapshot,
        e: &Entry,
        used: SystemTime,
        kept: bool,
    ) -> Option<String> {
        let len = e.tokens.len();
        let session = e.number;
        let decision = cache.save_decision(e, kept);
        let (target, from, version, mut parts, replaced, old) = match decision {
            Save::Skip => {
                return (len >= cache.disk_tokens && cache.disk_limit > 0).then(|| {
                    format!(
                        "session {session} ({len} tokens) not saved: other states hold all but fewer than {} of its tokens",
                        cache.disk_tokens
                    )
                });
            }
            Save::Append { saved, from } => {
                let d = &cache.saved[&saved];
                let replaced = d.replaced + d.tokens.len().saturating_sub(from);
                (saved, from, d.version + 1, d.parts.clone(), replaced, d.snaps.clone())
            }
            Save::New { .. } => {
                let id = self.next;
                self.next += 1;
                (id, 0, 1, Vec::new(), 0, Vec::new())
            }
        };
        // its rows from `from`, its newest prompt end and branch, and its end
        let newest = |k: Kind| e.snaps.iter().filter(|s| s.kind == k).max_by_key(|s| s.pos).copied();
        let mut snaps: Vec<(Snap, *mut RawSnapshot)> = [newest(Kind::PromptEnd), newest(Kind::Branch)]
            .into_iter()
            .flatten()
            .map(|s| (s, snapshot(s.pos)))
            .filter(|(_, x)| !x.is_null())
            .collect();
        if !snaps.iter().any(|(s, _)| s.pos == len) {
            let end = Snap {
                pos: len,
                kind: Kind::GenerationEnd,
                bytes: 0,
                stale: e.stale,
            };
            snaps.push((end, std::ptr::null_mut()));
        }
        let sizes: Option<Vec<u64>> = snaps.iter().map(|(s, x)| io.snapshot(*x, s.pos, -1)).collect();
        let (Some(rows), Some(sizes)) = (io.rows(from, len, -1), sizes) else {
            return Some(format!("session {session} not saved: the engine cannot write its state"));
        };
        // the room it takes: its files, less those that it replaces (an
        // append's old snapshots, a replaced saved state)
        let need = rows + sizes.iter().sum::<u64>();
        let freed = match decision {
            Save::New { replaces: Some(r) } => cache.saved.get(&r).map_or(0, Saved::bytes),
            _ => old.iter().map(|x| x.snap.bytes).sum(),
        };
        let keep = match decision {
            Save::New { replaces: Some(r) } => Some(r),
            _ => Some(target),
        };
        let Some(room) = cache.disk_room(need.saturating_sub(freed), keep, Some(e.used)) else {
            return Some(format!(
                "session {session} ({len} tokens, {:.2} GiB) not saved: kv_disk ({:.1} GiB) holds states used more recently",
                need as f64 / GIB,
                cache.disk_limit as f64 / GIB
            ));
        };
        if self.available() < need + VOLUME_RESERVE {
            return Some(format!(
                "session {session} ({len} tokens, {:.2} GiB) not saved: it would leave less than {} GiB free on the volume",
                need as f64 / GIB,
                VOLUME_RESERVE >> 30
            ));
        }
        let t = Instant::now();
        let file = parts.last().map_or(0, |p| p.file + 1);
        let written = self.write_file(&format!("{target}-{file}.rows"), |fd| io.rows(from, len, fd));
        let part = match written {
            Ok(bytes) => Part {
                from,
                to: len,
                bytes,
                file,
            },
            Err(m) => return Some(format!("session {session} not saved: {m}")),
        };
        let mut saved_snaps = Vec::new();
        for (s, x) in &snaps {
            let name = format!("{target}-{version}-{}.snap", s.pos);
            match self.write_file(&name, |fd| io.snapshot(*x, s.pos, fd)) {
                Ok(bytes) => saved_snaps.push(SavedSnap {
                    snap: Snap { bytes, ..*s },
                    version,
                }),
                Err(m) => {
                    self.remove_files(target, &[part], &saved_snaps);
                    return Some(format!("session {session} not saved: {m}"));
                }
            }
        }
        parts.push(part);
        let saved = Saved {
            model: Some(e.model),
            tokens: e.tokens.clone(),
            parts,
            snaps: saved_snaps,
            replaced,
            used: 0,
            version,
        };
        if let Err(m) = self.write_index(target, &io.identity(), &saved) {
            self.remove_files(target, &[part], &saved.snaps);
            return Some(format!("session {session} not saved: {m}"));
        }
        self.set_used(target, used);
        // once it is written: what the new index no longer names, what it
        // replaces, and the room it takes from older saved states
        self.remove_files(target, &[], &old);
        if let Save::New { replaces: Some(r) } = decision {
            self.forget(cache, r);
        }
        for id in room {
            self.forget(cache, id);
        }
        let bytes = part.bytes + saved.snaps.iter().map(|x| x.snap.bytes).sum::<u64>();
        cache.insert_saved(target, saved);
        if let Some(d) = cache.saved.get_mut(&target) {
            d.used = e.used; // the state's own last use, in the cache's order
        }
        for id in cache.covered(target) {
            self.forget(cache, id);
        }
        for id in cache.disk_room(0, Some(target), None).unwrap_or_default() {
            self.forget(cache, id);
        }
        self.saves += 1;
        self.written += bytes;
        Some(format!(
            "session {session} saved to the SSD as {target}: {len} tokens{}, {:.2} GiB written in {:.2} s; saved states take {:.1} of {:.1} GiB",
            if from > 0 { format!(" (from position {from})") } else { String::new() },
            bytes as f64 / GIB,
            t.elapsed().as_secs_f64(),
            cache.saved_bytes() as f64 / GIB,
            cache.disk_limit as f64 / GIB
        ))
    }

    /// Reads saved state `id` into a new state through `io`: its rows before
    /// `at` and, with `snap`, its snapshot there (returned, not restored); the
    /// saved state is used now.
    pub fn load(
        &mut self,
        cache: &mut Cache,
        io: &dyn StateIo,
        id: DiskId,
        snap: Option<usize>,
        at: usize,
    ) -> Result<*mut RawSnapshot, LoadError> {
        let damaged = |m: String| LoadError::Damaged(m);
        let d = cache.saved.get(&id).ok_or(damaged("it is gone".into()))?;
        // a missing file is damage; one that cannot be opened now (its
        // permissions, too many open files) is not
        let open = |name: String| {
            File::open(self.dir.join(&name)).map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => damaged(format!("{name}: {e}")),
                _ => LoadError::Unavailable(format!("{name}: {e}")),
            })
        };
        let files = d.parts.iter().map(|p| open(part_name(id, p))).collect::<Result<Vec<File>, LoadError>>()?;
        let fds: Vec<c_int> = files.iter().map(|f| f.as_raw_fd()).collect();
        if !io.read_rows(at, &fds) {
            return Err(damaged("its rows are damaged or of another shape".into()));
        }
        let mut x = std::ptr::null_mut();
        if let Some(pos) = snap {
            let s = d.snaps.iter().find(|s| s.snap.pos == pos).ok_or(damaged("its snapshot is gone".into()))?;
            x = io.read_snapshot(open(snap_name(id, s))?.as_raw_fd());
            if x.is_null() {
                return Err(damaged("its snapshot is damaged or of another shape".into()));
            }
        }
        self.set_used(id, SystemTime::now());
        cache.touch_saved(id);
        self.loads += 1;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::Link;

    fn saved(tokens: usize) -> Saved {
        Saved {
            model: None,
            tokens: (0..tokens as i32).map(|t| t * 7 - 3).collect(),
            parts: vec![
                Part { from: 0, to: tokens / 2, bytes: 1000, file: 0 },
                Part { from: tokens / 2 - 1, to: tokens, bytes: 500, file: 1 },
            ],
            snaps: vec![SavedSnap {
                snap: Snap { pos: tokens, kind: Kind::PromptEnd, bytes: 77, stale: 2 },
                version: 3,
            }],
            replaced: 1,
            used: 0,
            version: 3,
        }
    }

    #[test]
    fn an_index_reads_back_as_written_and_a_changed_byte_is_refused() {
        let s = saved(100);
        let id = [9u8; 32];
        let b = encode(&id, &s);
        let (identity, back) = decode(&b).expect("an index");
        assert_eq!(identity, id);
        assert_eq!((back.tokens, back.parts, back.snaps, back.replaced, back.version), (s.tokens, s.parts, s.snaps, 1, 3));
        for at in [0, 9, 20, 60, 110, b.len() - 1] {
            let mut bad = b.clone();
            bad[at] ^= 1;
            assert!(decode(&bad).is_none(), "a changed byte at {at}");
        }
        assert!(decode(&b[..b.len() - 4]).is_none(), "a short index");
    }

    #[test]
    fn an_index_whose_parts_do_not_continue_each_other_is_refused() {
        let mut s = saved(100);
        s.parts[1].from = 60; // a gap after the first part's end
        assert!(decode(&encode(&[0; 32], &s)).is_none());
        let mut s = saved(100);
        s.parts[1].to = 99; // short of the state's end
        assert!(decode(&encode(&[0; 32], &s)).is_none());
    }

    #[test]
    fn only_the_stores_names_are_its_files() {
        assert_eq!(parse_name("12.kv"), Some((Name::Index(12), false)));
        assert_eq!(parse_name("12-3.rows"), Some((Name::Part(12), false)));
        assert_eq!(parse_name("12-3-4096.snap.tmp"), Some((Name::Snap(12), true)));
        for other in ["lock", "notes.txt", "12.kv.bak", "a-3.rows", "12-3.snap", "12-x-4.snap", ".DS_Store", "12.KV"] {
            assert_eq!(parse_name(other), None, "{other}");
        }
        // numbers the store never writes: with leading zeros
        for other in ["05.kv", "05-0.rows", "1-00.rows", "1-1-007.snap", "007-1-2.snap.tmp"] {
            assert_eq!(parse_name(other), None, "{other}");
        }
    }

    /// A state whose rows and snapshots are `rows` and `snap` zero bytes,
    /// written or (with `fail`) not.
    struct Fake {
        rows: u64,
        snap: u64,
        fail: bool,
    }

    impl Fake {
        fn write(&self, n: u64, fd: c_int) -> Option<u64> {
            if fd < 0 {
                return Some(n);
            }
            if self.fail {
                return None;
            }
            let mut file = std::mem::ManuallyDrop::new(unsafe { <File as std::os::fd::FromRawFd>::from_raw_fd(fd) });
            std::io::Write::write_all(&mut *file, &vec![0u8; n as usize]).ok().map(|_| n)
        }
    }

    impl StateIo for Fake {
        fn rows(&self, _from: usize, _to: usize, fd: c_int) -> Option<u64> {
            self.write(self.rows, fd)
        }
        fn snapshot(&self, _x: *const RawSnapshot, _pos: usize, fd: c_int) -> Option<u64> {
            self.write(self.snap, fd)
        }
        fn identity(&self) -> [u8; 32] {
            [3; 32]
        }
        fn read_rows(&self, _positions: usize, _fds: &[c_int]) -> bool {
            true
        }
        fn read_snapshot(&self, _fd: c_int) -> *mut RawSnapshot {
            std::ptr::NonNull::dangling().as_ptr()
        }
    }

    /// A store in a new folder, and a cache that saves states of 100 tokens
    /// of their own.
    fn store(name: &str) -> (PathBuf, Store, Cache) {
        let dir = std::env::temp_dir().join(format!("myds4-disk-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let (store, _) = Store::open(&dir).expect("the store");
        let mut cache = Cache::new(1 << 30, 16);
        cache.disk_tokens = 100;
        cache.disk_limit = 1 << 30;
        (dir, store, cache)
    }

    /// Saved state `id` of model 0 holding `tokens`, a part of `rows` bytes
    /// and an end snapshot of `snap` bytes, its files written.
    fn put(dir: &Path, cache: &mut Cache, id: DiskId, tokens: std::ops::Range<i32>, rows: u64, snap: u64) {
        let len = tokens.len();
        let saved = Saved {
            model: Some(0),
            tokens: tokens.collect(),
            parts: vec![Part { from: 0, to: len, bytes: rows, file: 0 }],
            snaps: vec![SavedSnap {
                snap: Snap { pos: len, kind: Kind::GenerationEnd, bytes: snap, stale: 0 },
                version: 1,
            }],
            replaced: 0,
            used: 0,
            version: 1,
        };
        fs::write(dir.join(format!("{id}.kv")), encode(&[3; 32], &saved)).unwrap();
        fs::write(dir.join(part_name(id, &saved.parts[0])), vec![0u8; rows as usize]).unwrap();
        fs::write(dir.join(snap_name(id, &saved.snaps[0])), vec![0u8; snap as usize]).unwrap();
        cache.insert_saved(id, saved);
    }

    fn entry(tokens: std::ops::Range<i32>, used: u64) -> Entry {
        Entry {
            model: 0,
            tokens: tokens.collect(),
            busy: false,
            used,
            bytes: 0,
            stale: 0,
            snaps: Vec::new(),
            pending: Vec::new(),
            number: 1,
            hits: Vec::new(),
            saved: None,
        }
    }

    #[test]
    fn a_failed_save_takes_no_room_from_other_saved_states() {
        let (dir, mut store, mut cache) = store("failed-save");
        put(&dir, &mut cache, 7, 5000..5400, 800, 200); // used before the state
        cache.disk_limit = 1100;
        let e = entry(0..500, 100);
        let failing = Fake { rows: 300, snap: 500, fail: true };
        let line = store.save(&mut cache, &failing, &|_| std::ptr::null_mut(), &e, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("not saved"));
        assert!(cache.saved.contains_key(&7) && dir.join("7.kv").exists(), "the older saved state stays");
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_append_and_a_replacement_count_what_they_replace_as_room() {
        let (dir, mut store, mut cache) = store("room");
        put(&dir, &mut cache, 4, 0..1000, 1000, 500);
        put(&dir, &mut cache, 5, 5000..5400, 800, 200); // used before the states saved below
        // the new part fits once the saved state's old end snapshot goes
        cache.disk_limit = 1500 + 1000 + 300;
        let mut e = entry(0..1300, 100);
        e.saved = Some(Link { saved: 4, version: 1, shared: 1000 });
        let io = Fake { rows: 300, snap: 500, fail: false };
        let line = store.save(&mut cache, &io, &|_| std::ptr::null_mut(), &e, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("saved to the SSD as 4"));
        assert!(cache.saved.contains_key(&5), "the other saved state stays");
        assert_eq!(cache.saved_bytes(), 1000 + 300 + 500 + 1000);
        assert!(!dir.join("4-1-1000.snap").exists(), "the old end snapshot went");
        // written whole once its replaced rows outnumber its own: in its own room
        cache.saved.get_mut(&4).unwrap().replaced = 5000;
        let mut e = entry(0..1500, 200);
        e.saved = Some(Link { saved: 4, version: 2, shared: 1300 });
        let line = store.save(&mut cache, &io, &|_| std::ptr::null_mut(), &e, SystemTime::now(), false);
        assert!(line.unwrap_or_default().contains("saved to the SSD"));
        assert!(cache.saved.contains_key(&5) && !cache.saved.contains_key(&4), "it replaced 4, not 5");
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_saved_state_that_cannot_be_opened_now_is_not_damaged() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, mut store, mut cache) = store("load-errors");
        put(&dir, &mut cache, 4, 0..1000, 1000, 500);
        let io = Fake { rows: 0, snap: 0, fail: false };
        let part = dir.join("4-0.rows");
        fs::set_permissions(&part, fs::Permissions::from_mode(0o000)).unwrap();
        let read = store.load(&mut cache, &io, 4, Some(1000), 1000);
        fs::set_permissions(&part, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(matches!(read, Err(LoadError::Unavailable(_))), "{read:?}");
        assert!(store.load(&mut cache, &io, 4, Some(1000), 1000).is_ok());
        fs::remove_file(&part).unwrap();
        let read = store.load(&mut cache, &io, 4, Some(1000), 1000);
        assert!(matches!(read, Err(LoadError::Damaged(_))), "a missing part: {read:?}");
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_index_of_another_version_is_kept_with_its_files() {
        let dir = std::env::temp_dir().join(format!("myds4-disk-version-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut index = encode(&[1; 32], &saved(100));
        index[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes()); // a later server's
        fs::write(dir.join("9.kv"), &index).unwrap();
        fs::write(dir.join("9-0.rows"), vec![0u8; 1000]).unwrap();
        fs::write(dir.join("9-7-100.snap"), vec![0u8; 77]).unwrap();
        let (store, found) = Store::open(&dir).expect("the store");
        assert!(found.is_empty(), "this server does not read it");
        assert!(store.next >= 10, "nor reuses its number");
        for name in ["9.kv", "9-0.rows", "9-7-100.snap"] {
            assert!(dir.join(name).exists(), "{name} is kept");
        }
        drop(store);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_store_keeps_named_files_deletes_its_orphans_and_leaves_others() {
        let dir = std::env::temp_dir().join(format!("myds4-disk-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let s = saved(100);
        fs::write(dir.join("5.kv"), encode(&[1; 32], &s)).unwrap();
        fs::write(dir.join("5-0.rows"), vec![0u8; 1000]).unwrap();
        fs::write(dir.join("5-1.rows"), vec![0u8; 500]).unwrap();
        fs::write(dir.join("5-3-100.snap"), vec![0u8; 77]).unwrap();
        fs::write(dir.join("5-2-100.snap"), b"an older save's").unwrap(); // named by no index
        fs::write(dir.join("6-0.rows.tmp"), b"a crash's").unwrap();
        fs::write(dir.join("7.kv"), b"damaged").unwrap();
        fs::write(dir.join("notes.txt"), b"the user's").unwrap();
        {
            let (store, found) = Store::open(&dir).expect("the store");
            assert_eq!(found.len(), 1);
            assert_eq!((found[0].id, found[0].identity), (5, [1; 32]));
            assert!(store.next >= 8);
            assert!(Store::open(&dir).is_err(), "a second server is refused while the first holds the lock");
        }
        let mut names: Vec<String> = fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["5-0.rows", "5-1.rows", "5-3-100.snap", "5.kv", "lock", "notes.txt"]);
        // a part shorter than its index says: the saved state goes
        fs::write(dir.join("5-1.rows"), vec![0u8; 499]).unwrap();
        let (_, found) = Store::open(&dir).expect("the store");
        assert!(found.is_empty());
        assert!(!dir.join("5.kv").exists() && !dir.join("5-0.rows").exists());
        assert!(dir.join("notes.txt").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
