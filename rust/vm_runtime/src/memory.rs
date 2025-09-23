//! Memory subsystem for the Praxis VM runtime.
//!
//! The runtime intentionally favours safety and observability over raw
//! performance: values are immutable by default, container growth is
//! accounted-for explicitly, and all heap activity flows through this module so
//! that the sandbox can enforce global limits.  The design keeps allocations
//! lightweight for common scalar cases while providing coarse tracking for
//! container-heavy workloads.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bytecode::ScalarValue;

/// Default heap ceiling applied when the caller does not provide an explicit
/// limit (16 MiB keeps fixture programs honest yet generous).
pub const DEFAULT_HEAP_LIMIT_BYTES: usize = 16 * 1024 * 1024;

/// Approximate accounting constants used when tracking container growth.
const CONTAINER_BASE_BYTES: usize = 64;
const LIST_ELEMENT_BYTES: usize = 16;
const MAP_ENTRY_BYTES: usize = 48;
const SET_ENTRY_BYTES: usize = 32;

/// Runtime value stored inside registers, the operand stack, and containers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    Scalar(ScalarValue),
    List(Vec<Value>),
    Map(BTreeMap<ValueKey, Value>),
    Set(BTreeSet<ValueKey>),
}

impl Value {
    /// Helpful string describing the variant for diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Scalar(ScalarValue::Unit) => "unit",
            Value::Scalar(ScalarValue::Int(_)) => "int",
            Value::Scalar(ScalarValue::Float(_)) => "float",
            Value::Scalar(ScalarValue::Bool(_)) => "bool",
            Value::Scalar(ScalarValue::String(_)) => "string",
            Value::List(_) => "list",
            Value::Map(_) => "map",
            Value::Set(_) => "set",
        }
    }

    pub fn unit() -> Self {
        Value::Scalar(ScalarValue::Unit)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Scalar(ScalarValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Scalar(ScalarValue::Int(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Scalar(ScalarValue::Float(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::Scalar(ScalarValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Return the number of elements held by the container value.
    pub fn len(&self) -> Option<usize> {
        match self {
            Value::List(items) => Some(items.len()),
            Value::Map(entries) => Some(entries.len()),
            Value::Set(entries) => Some(entries.len()),
            Value::Scalar(ScalarValue::String(value)) => Some(value.chars().count()),
            _ => None,
        }
    }

    /// Indicates whether the container is empty. Mirrors [`Self::len`].
    pub fn is_empty(&self) -> Option<bool> {
        match self {
            Value::List(items) => Some(items.is_empty()),
            Value::Map(entries) => Some(entries.is_empty()),
            Value::Set(entries) => Some(entries.is_empty()),
            Value::Scalar(ScalarValue::String(value)) => Some(value.is_empty()),
            _ => None,
        }
    }
}

impl From<ScalarValue> for Value {
    fn from(value: ScalarValue) -> Self {
        Value::Scalar(value)
    }
}

/// Map/set key representation compatible with the serialised `Value` schema.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ValueKey {
    Int(i64),
    String(String),
    Bool(bool),
}

impl<'a> TryFrom<&'a Value> for ValueKey {
    type Error = MemoryError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Scalar(ScalarValue::Int(inner)) => Ok(ValueKey::Int(*inner)),
            Value::Scalar(ScalarValue::String(inner)) => Ok(ValueKey::String(inner.clone())),
            Value::Scalar(ScalarValue::Bool(inner)) => Ok(ValueKey::Bool(*inner)),
            other => Err(MemoryError::InvalidKey(other.kind())),
        }
    }
}

/// Memory usage snapshot emitted as part of the execution trace.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub bytes_current: usize,
    pub bytes_peak: usize,
    pub allocations: usize,
}

/// Heap manager enforcing conservative container accounting.
#[derive(Debug, Clone)]
pub struct Heap {
    limit_bytes: usize,
    metrics: MemoryMetrics,
}

impl Heap {
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            limit_bytes,
            metrics: MemoryMetrics::default(),
        }
    }

    pub fn with_default_limit() -> Self {
        Self::new(DEFAULT_HEAP_LIMIT_BYTES)
    }

    pub fn metrics(&self) -> &MemoryMetrics {
        &self.metrics
    }

    #[inline]
    pub fn limit(&self) -> usize {
        self.limit_bytes
    }

    /// Reserve space for a container that is about to be created.
    pub fn register_container_allocation(&mut self) -> Result<(), MemoryError> {
        self.metrics.allocations += 1;
        self.grow(CONTAINER_BASE_BYTES)
    }

    /// Account for a single element inserted into a list.
    pub fn list_grew(&mut self) -> Result<(), MemoryError> {
        self.grow(LIST_ELEMENT_BYTES)
    }

    /// Account for an element removed from a list.
    pub fn list_shrank(&mut self) {
        self.shrink(LIST_ELEMENT_BYTES)
    }

    /// Account for a map insertion (key/value pair).
    pub fn map_grew(&mut self) -> Result<(), MemoryError> {
        self.grow(MAP_ENTRY_BYTES)
    }

    /// Account for a map entry removal.
    pub fn map_shrank(&mut self) {
        self.shrink(MAP_ENTRY_BYTES)
    }

    /// Account for adding a set entry.
    pub fn set_grew(&mut self) -> Result<(), MemoryError> {
        self.grow(SET_ENTRY_BYTES)
    }

    /// Account for removing a set entry.
    pub fn set_shrank(&mut self) {
        self.shrink(SET_ENTRY_BYTES)
    }

    fn grow(&mut self, bytes: usize) -> Result<(), MemoryError> {
        if bytes == 0 {
            return Ok(());
        }
        let projected = self.metrics.bytes_current.saturating_add(bytes);
        if projected > self.limit_bytes {
            return Err(MemoryError::LimitExceeded {
                used: self.metrics.bytes_current,
                requested: bytes,
                limit: self.limit_bytes,
            });
        }
        self.metrics.bytes_current = projected;
        if projected > self.metrics.bytes_peak {
            self.metrics.bytes_peak = projected;
        }
        Ok(())
    }

    fn shrink(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.metrics.bytes_current = self.metrics.bytes_current.saturating_sub(bytes);
    }
}

/// Memory level errors bubbled up to the interpreter.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("heap limit exceeded: used {used} bytes, attempted {requested} additional bytes (limit {limit})")]
    LimitExceeded {
        used: usize,
        requested: usize,
        limit: usize,
    },
    #[error("invalid key type for map/set operation: {0}")]
    InvalidKey(&'static str),
}
