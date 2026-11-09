//! Process-wide yield ownership and temporary memory limits. The worker
//! completes transitions after draining the engine and reclaiming memory.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YieldRequest {
    pub operation_id: String,
    pub memory_gib: Option<f64>,
    pub kv_cache_gib: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeRequest {
    pub operation_id: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Limits {
    pub model_bytes: u64,
    pub kv_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Minimum,
    Bytes(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Targets {
    pub model: Target,
    pub kv: Target,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Running,
    Yielding,
    Yielded,
    Resuming,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Yielding => "yielding",
            Self::Yielded => "yielded",
            Self::Resuming => "resuming",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl Error {
    fn invalid(message: &str) -> Self {
        Self {
            status: 400,
            code: "invalid_control_request",
            message: message.into(),
        }
    }
    fn conflict(code: &'static str, message: &str) -> Self {
        Self {
            status: 409,
            code,
            message: message.into(),
        }
    }
}

fn operation_id(id: &str) -> Result<(), Error> {
    if id.is_empty() || id.len() > 128 || id.trim().is_empty() || id.chars().any(char::is_control) {
        return Err(Error::invalid(
            "operation_id must contain 1 to 128 bytes without control characters",
        ));
    }
    Ok(())
}

fn target(value: Option<f64>, configured: u64) -> Result<Target, Error> {
    let Some(gib) = value else {
        return Ok(Target::Bytes(configured));
    };
    if !gib.is_finite() || gib < 0.0 || gib > configured as f64 / GIB {
        return Err(Error::invalid(
            "memory limits must be finite, nonnegative and no larger than their configured limits",
        ));
    }
    Ok(if gib == 0.0 {
        Target::Minimum
    } else {
        Target::Bytes((gib * GIB) as u64)
    })
}

enum Outcome {
    Active,
    Resumed,
    Failed(String),
}
struct Record {
    targets: Targets,
    outcome: Outcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Yield(Targets),
    Resume,
}

pub struct Control {
    pub phase: Phase,
    pub configured: Limits,
    pub effective: Limits,
    pub retained: Limits,
    owner: Option<String>,
    records: HashMap<String, Record>,
    executing: bool,
    failure: Option<Value>,
}

impl Default for Control {
    fn default() -> Self {
        Self::new(Limits::default())
    }
}

impl Control {
    pub fn new(configured: Limits) -> Self {
        Self {
            phase: Phase::Running,
            configured,
            effective: configured,
            retained: Limits::default(),
            owner: None,
            records: HashMap::new(),
            executing: false,
            failure: None,
        }
    }

    pub fn snapshot(&self) -> Value {
        json!({"state": self.phase, "operation_id": self.owner,
            "configured_limits": self.configured, "effective_limits": self.effective,
            "retained": self.retained, "error": self.failure})
    }

    pub fn yield_request(&mut self, request: YieldRequest) -> Result<u16, Error> {
        operation_id(&request.operation_id)?;
        let targets = Targets {
            model: target(request.memory_gib, self.configured.model_bytes)?,
            kv: target(request.kv_cache_gib, self.configured.kv_bytes)?,
        };
        if let Some(record) = self.records.get(&request.operation_id) {
            if record.targets != targets {
                return Err(Error::conflict(
                    "operation_conflict",
                    "this operation_id was used with different limits",
                ));
            }
            return match &record.outcome {
                Outcome::Active => Ok(if self.phase == Phase::Yielded {
                    200
                } else {
                    202
                }),
                Outcome::Resumed => Err(Error::conflict(
                    "operation_completed",
                    "this operation was already resumed; use a new operation_id",
                )),
                Outcome::Failed(message) => Err(Error::conflict("yield_failed", message)),
            };
        }
        if self.owner.is_some() {
            return Err(Error::conflict(
                "operation_conflict",
                "another yield operation owns the server",
            ));
        }
        self.records.insert(
            request.operation_id.clone(),
            Record {
                targets,
                outcome: Outcome::Active,
            },
        );
        self.owner = Some(request.operation_id);
        self.phase = Phase::Yielding;
        self.failure = None;
        Ok(202)
    }

    /// True means this operation already ended; it must not resume a later owner.
    pub fn resume_request(&mut self, id: &str) -> Result<bool, Error> {
        operation_id(id)?;
        match self.records.get(id).map(|record| &record.outcome) {
            Some(Outcome::Resumed | Outcome::Failed(_)) => return Ok(true),
            Some(Outcome::Active) => {}
            None => {
                return Err(Error::conflict(
                    "operation_conflict",
                    "operation_id does not own a yield",
                ));
            }
        }
        self.phase = Phase::Resuming;
        Ok(false)
    }

    pub fn next_action(&mut self) -> Option<Action> {
        if self.executing {
            return None;
        }
        let action = match self.phase {
            Phase::Yielding => Action::Yield(self.records[self.owner.as_ref().unwrap()].targets),
            Phase::Resuming => Action::Resume,
            Phase::Running | Phase::Yielded => return None,
        };
        self.executing = true;
        Some(action)
    }

    pub fn yielded(&mut self, limits: Limits, retained: Limits) {
        self.effective = limits;
        self.retained = retained;
        self.executing = false;
        if self.phase != Phase::Resuming {
            self.phase = Phase::Yielded;
        }
    }

    pub fn failed(&mut self, message: String, retained: Limits) {
        let id = self.owner.take().unwrap();
        self.records.get_mut(&id).unwrap().outcome = Outcome::Failed(message.clone());
        self.failure =
            Some(json!({"operation_id":id, "code":"limits_unreachable", "message":message}));
        self.phase = Phase::Running;
        self.effective = self.configured;
        self.retained = retained;
        self.executing = false;
    }

    pub fn resumed(&mut self, retained: Limits) {
        let id = self.owner.take().unwrap();
        self.records.get_mut(&id).unwrap().outcome = Outcome::Resumed;
        self.phase = Phase::Running;
        self.effective = self.configured;
        self.retained = retained;
        self.executing = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn control() -> Control {
        Control::new(Limits {
            model_bytes: 64 << 30,
            kv_bytes: 8 << 30,
        })
    }
    fn request(id: &str) -> YieldRequest {
        YieldRequest {
            operation_id: id.into(),
            memory_gib: Some(4.0),
            kv_cache_gib: Some(2.0),
        }
    }
    #[test]
    fn yields_are_pending_until_the_worker_confirms_drain_and_reclaim() {
        let mut c = control();
        assert_eq!(c.yield_request(request("a")).unwrap(), 202);
        assert_eq!(c.phase, Phase::Yielding);
        assert_eq!(c.effective, c.configured);
        assert!(matches!(c.next_action(), Some(Action::Yield(_))));
        assert_eq!(c.next_action(), None);
        let limits = Limits {
            model_bytes: 4 << 30,
            kv_bytes: 2 << 30,
        };
        c.yielded(limits, limits);
        assert_eq!(c.phase, Phase::Yielded);
        assert_eq!(c.yield_request(request("a")).unwrap(), 200);
        assert_eq!(c.next_action(), None);
    }
    #[test]
    fn conflicting_calls_leave_the_current_owner_unchanged() {
        let mut c = control();
        c.yield_request(request("a")).unwrap();
        let before = c.snapshot();
        assert!(c.yield_request(request("b")).is_err());
        assert!(c.resume_request("b").is_err());
        let mut changed = request("a");
        changed.memory_gib = Some(3.0);
        assert!(c.yield_request(changed).is_err());
        assert_eq!(before, c.snapshot());
    }
    #[test]
    fn resume_during_yield_waits_for_the_in_flight_action() {
        let mut c = control();
        c.yield_request(request("a")).unwrap();
        c.next_action();
        assert!(!c.resume_request("a").unwrap());
        assert_eq!(c.next_action(), None);
        c.yielded(Limits::default(), Limits::default());
        assert_eq!(c.phase, Phase::Resuming);
        assert_eq!(c.next_action(), Some(Action::Resume));
        c.resumed(Limits::default());
        assert_eq!(c.phase, Phase::Running);
        assert_eq!(c.effective, c.configured);
    }
    #[test]
    fn completed_retries_never_yield_again_or_resume_a_new_owner() {
        let mut c = control();
        c.yield_request(request("a")).unwrap();
        c.resume_request("a").unwrap();
        c.next_action();
        c.resumed(Limits::default());
        assert!(c.yield_request(request("a")).is_err());
        c.yield_request(request("b")).unwrap();
        let before = c.snapshot();
        assert!(c.resume_request("a").unwrap());
        assert_eq!(before, c.snapshot());
    }
    #[test]
    fn omitted_limits_keep_the_budget_and_zero_requests_the_minimum() {
        let mut c = control();
        c.yield_request(YieldRequest {
            operation_id: "a".into(),
            memory_gib: None,
            kv_cache_gib: Some(0.0),
        })
        .unwrap();
        assert_eq!(
            c.next_action(),
            Some(Action::Yield(Targets {
                model: Target::Bytes(64 << 30),
                kv: Target::Minimum
            }))
        );
    }
    #[test]
    fn failed_yields_are_queryable_and_do_not_reapply_on_retry() {
        let mut c = control();
        c.yield_request(request("a")).unwrap();
        c.next_action();
        c.failed("active KV does not fit".into(), Limits::default());
        assert_eq!(c.phase, Phase::Running);
        assert_eq!(c.snapshot()["error"]["operation_id"], "a");
        assert!(c.yield_request(request("a")).is_err());
        assert!(c.resume_request("a").unwrap());
    }
    #[test]
    fn invalid_input_is_rejected_at_the_control_boundary() {
        for value in [-1.0, 65.0, f64::INFINITY, f64::NAN] {
            let mut c = control();
            let mut r = request("a");
            r.memory_gib = Some(value);
            assert_eq!(c.yield_request(r).unwrap_err().status, 400);
            assert_eq!(c.phase, Phase::Running);
        }
        for id in ["", " \t", "bad\nline"] {
            assert!(control().yield_request(request(id)).is_err());
        }
    }
}
