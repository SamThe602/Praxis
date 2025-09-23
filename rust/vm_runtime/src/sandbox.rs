//! Execution sandbox enforcing deterministic resource limits.
//!
//! The VM is designed to run untrusted bytecode emitted by the synthesis
//! pipeline.  To keep runs both predictable and safe we impose conservative
//! bounds on instructions executed, stack growth, call depth, wall-clock time
//! and heap usage.  The heap checks are performed cooperatively with the memory
//! subsystem while the other limits are tracked directly here.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::memory::DEFAULT_HEAP_LIMIT_BYTES;

mod duration_format {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

/// Call-site configurable sandbox limits.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SandboxLimits {
    pub instruction_limit: u64,
    pub stack_limit: usize,
    pub call_depth_limit: usize,
    #[serde(with = "duration_format")]
    pub wall_time: Duration,
    pub heap_limit_bytes: usize,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            instruction_limit: 100_000,
            stack_limit: 1_024,
            call_depth_limit: 128,
            wall_time: Duration::from_millis(200),
            heap_limit_bytes: DEFAULT_HEAP_LIMIT_BYTES,
        }
    }
}

/// Minimal set of counters surfaced for observability.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SandboxMetrics {
    pub instruction_count: u64,
    pub max_stack_depth: usize,
    pub max_call_depth: usize,
    #[serde(with = "duration_format")]
    pub elapsed: Duration,
}

/// Stateful sandbox tracker used by the interpreter.
#[derive(Debug)]
pub struct Sandbox {
    limits: SandboxLimits,
    start: Instant,
    metrics: SandboxMetrics,
}

impl Sandbox {
    pub fn new(limits: SandboxLimits) -> Self {
        Self {
            limits,
            start: Instant::now(),
            metrics: SandboxMetrics::default(),
        }
    }

    pub fn limits(&self) -> SandboxLimits {
        self.limits
    }

    pub fn metrics(&self) -> SandboxMetrics {
        let mut metrics = self.metrics;
        metrics.elapsed = self.start.elapsed();
        metrics
    }

    /// Call before executing each instruction.
    pub fn observe_instruction(&mut self) -> Result<(), SandboxError> {
        self.metrics.instruction_count += 1;
        if self.metrics.instruction_count > self.limits.instruction_limit {
            return Err(SandboxError::InstructionLimit {
                limit: self.limits.instruction_limit,
            });
        }
        self.enforce_wall_time()
    }

    /// Update stack depth accounting and enforce limits.
    pub fn record_stack_depth(&mut self, depth: usize) -> Result<(), SandboxError> {
        if depth > self.metrics.max_stack_depth {
            self.metrics.max_stack_depth = depth;
        }
        if depth > self.limits.stack_limit {
            return Err(SandboxError::StackLimit {
                limit: self.limits.stack_limit,
            });
        }
        Ok(())
    }

    /// Update call-depth accounting and enforce limits.
    pub fn record_call_depth(&mut self, depth: usize) -> Result<(), SandboxError> {
        if depth > self.metrics.max_call_depth {
            self.metrics.max_call_depth = depth;
        }
        if depth > self.limits.call_depth_limit {
            return Err(SandboxError::CallDepthLimit {
                limit: self.limits.call_depth_limit,
            });
        }
        Ok(())
    }

    pub fn enforce_heap_usage(&self, bytes: usize) -> Result<(), SandboxError> {
        if bytes > self.limits.heap_limit_bytes {
            return Err(SandboxError::HeapLimit {
                limit: self.limits.heap_limit_bytes,
                used: bytes,
            });
        }
        Ok(())
    }

    fn enforce_wall_time(&self) -> Result<(), SandboxError> {
        if self.limits.wall_time.is_zero() {
            return Ok(());
        }
        if self.start.elapsed() > self.limits.wall_time {
            return Err(SandboxError::Timeout {
                limit: self.limits.wall_time,
            });
        }
        Ok(())
    }
}

/// Sandbox enforcement errors raised to the caller.
#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("instruction limit exceeded (limit {limit})")]
    InstructionLimit { limit: u64 },
    #[error("stack depth limit exceeded (limit {limit})")]
    StackLimit { limit: usize },
    #[error("call depth limit exceeded (limit {limit})")]
    CallDepthLimit { limit: usize },
    #[error("execution timed out after {limit:?}")]
    Timeout { limit: Duration },
    #[error("heap usage {used} bytes exceeds sandbox limit {limit} bytes")]
    HeapLimit { limit: usize, used: usize },
}
