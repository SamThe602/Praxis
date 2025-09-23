//! Built-in library providing deterministic host functions for the VM.
//!
//! The Praxis bytecode VM intentionally keeps its built-in surface area small
//! and pure so that execution remains reproducible.  Each builtin takes a slice
//! of already-evaluated arguments and returns either a value or an error that is
//! bubbled up through the interpreter.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bytecode::ScalarValue;
use crate::memory::Value;

/// Runtime error surfaced when a builtin rejects provided arguments.
#[derive(Debug, Error)]
pub enum BuiltinError {
    #[error("unknown intrinsic '{0}'")]
    Unknown(String),
    #[error("expected {expected} arguments but received {actual}")]
    Arity { expected: usize, actual: usize },
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        expected: &'static str,
        found: &'static str,
    },
    #[error("built-in computation failed: {0}")]
    InvalidState(&'static str),
}

pub type BuiltinResult = Result<Value, BuiltinError>;

type BuiltinFn = fn(&[Value]) -> BuiltinResult;

/// Registry of available built-in intrinsics.
#[derive(Debug, Clone)]
pub struct Builtins {
    registry: HashMap<String, BuiltinFn>,
}

impl Builtins {
    /// Construct the default registry used by the interpreter.
    pub fn standard() -> Self {
        let mut registry: HashMap<String, BuiltinFn> = HashMap::new();
        registry.insert("identity".to_string(), builtin_identity as BuiltinFn);
        registry.insert("len".to_string(), builtin_len as BuiltinFn);
        registry.insert("sum".to_string(), builtin_sum as BuiltinFn);
        registry.insert("min".to_string(), builtin_min as BuiltinFn);
        registry.insert("max".to_string(), builtin_max as BuiltinFn);
        registry.insert("abs".to_string(), builtin_abs as BuiltinFn);
        registry.insert("not".to_string(), builtin_not as BuiltinFn);
        Self { registry }
    }

    pub fn call(&self, name: &str, args: &[Value]) -> BuiltinResult {
        let handler = self
            .registry
            .get(name)
            .ok_or_else(|| BuiltinError::Unknown(name.to_owned()))?;
        handler(args)
    }
}

fn ensure_arity(args: &[Value], expected: usize) -> Result<(), BuiltinError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(BuiltinError::Arity {
            expected,
            actual: args.len(),
        })
    }
}

fn builtin_identity(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    Ok(args[0].clone())
}

fn builtin_len(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    match &args[0] {
        Value::List(items) => Ok(ScalarValue::Int(items.len() as i64).into()),
        Value::Map(entries) => Ok(ScalarValue::Int(entries.len() as i64).into()),
        Value::Set(entries) => Ok(ScalarValue::Int(entries.len() as i64).into()),
        Value::Scalar(ScalarValue::String(text)) => {
            Ok(ScalarValue::Int(text.chars().count() as i64).into())
        }
        other => Err(BuiltinError::TypeMismatch {
            expected: "list|map|set|string",
            found: other.kind(),
        }),
    }
}

fn builtin_sum(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    match &args[0] {
        Value::List(items) => {
            let mut acc: i64 = 0;
            for value in items {
                match value {
                    Value::Scalar(ScalarValue::Int(num)) => acc += num,
                    other => {
                        return Err(BuiltinError::TypeMismatch {
                            expected: "list<int>",
                            found: other.kind(),
                        })
                    }
                }
            }
            Ok(ScalarValue::Int(acc).into())
        }
        other => Err(BuiltinError::TypeMismatch {
            expected: "list<int>",
            found: other.kind(),
        }),
    }
}

fn builtin_min(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    fold_min_max(args, FoldKind::Min)
}

fn builtin_max(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    fold_min_max(args, FoldKind::Max)
}

fn builtin_abs(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    match &args[0] {
        Value::Scalar(ScalarValue::Int(value)) => Ok(ScalarValue::Int(value.abs()).into()),
        Value::Scalar(ScalarValue::Float(value)) => Ok(ScalarValue::Float(value.abs()).into()),
        other => Err(BuiltinError::TypeMismatch {
            expected: "int|float",
            found: other.kind(),
        }),
    }
}

fn builtin_not(args: &[Value]) -> BuiltinResult {
    ensure_arity(args, 1)?;
    match &args[0] {
        Value::Scalar(ScalarValue::Bool(value)) => Ok(ScalarValue::Bool(!value).into()),
        other => Err(BuiltinError::TypeMismatch {
            expected: "bool",
            found: other.kind(),
        }),
    }
}

#[derive(Clone, Copy)]
enum FoldKind {
    Min,
    Max,
}

fn fold_min_max(args: &[Value], kind: FoldKind) -> BuiltinResult {
    match &args[0] {
        Value::List(items) if !items.is_empty() => {
            let mut it = items.iter();
            let first = it.next().expect("non-empty by guard");
            let mut acc = match first {
                Value::Scalar(ScalarValue::Int(num)) => *num,
                other => {
                    return Err(BuiltinError::TypeMismatch {
                        expected: "list<int>",
                        found: other.kind(),
                    })
                }
            };
            for value in it {
                acc = match value {
                    Value::Scalar(ScalarValue::Int(num)) => match kind {
                        FoldKind::Min => acc.min(*num),
                        FoldKind::Max => acc.max(*num),
                    },
                    other => {
                        return Err(BuiltinError::TypeMismatch {
                            expected: "list<int>",
                            found: other.kind(),
                        })
                    }
                };
            }
            Ok(ScalarValue::Int(acc).into())
        }
        Value::List(_) => Err(BuiltinError::InvalidState("min/max on empty list")),
        other => Err(BuiltinError::TypeMismatch {
            expected: "list<int>",
            found: other.kind(),
        }),
    }
}

/// Lightweight serialisable descriptor surfaced through the FFI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinDescriptor {
    pub name: String,
}

impl Builtins {
    /// Return the list of registered names for documentation and telemetry.
    pub fn descriptors(&self) -> Vec<BuiltinDescriptor> {
        self.registry
            .keys()
            .cloned()
            .map(|name| BuiltinDescriptor { name })
            .collect()
    }
}
