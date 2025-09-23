//! High-performance property checkers shared with the Python verifier.
//!
//! The helpers favour predictable control-flow and avoid allocations in the
//! common path so they can be invoked frequently from the synthesiser.

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// Result for a checker invocation mirroring [`super::metamorphic::RelationResult`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckerResult {
    pub ok: bool,
    pub message: Option<String>,
}

impl CheckerResult {
    pub const fn passed() -> Self {
        Self {
            ok: true,
            message: None,
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: Some(message.into()),
        }
    }
}

/// Ensures a slice is sorted in non-decreasing order.
pub fn sorted<T>(values: &[T]) -> CheckerResult
where
    T: Ord + Debug,
{
    for (index, window) in values.windows(2).enumerate() {
        if window[0] > window[1] {
            return CheckerResult::failed(format!(
                "sequence not sorted at indices {} and {}: {:?} then {:?}",
                index,
                index + 1,
                window[0],
                window[1]
            ));
        }
    }
    CheckerResult::passed()
}

/// Checks that a slice contains no duplicates.
pub fn unique<T>(values: &[T]) -> CheckerResult
where
    T: Eq + Hash + Debug,
{
    let mut seen = HashSet::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        if !seen.insert(value) {
            return CheckerResult::failed(format!(
                "duplicate element {:?} at index {}",
                value, index
            ));
        }
    }
    CheckerResult::passed()
}

/// Validates that the frequency histogram matches the expected counts.
pub fn histogram_matches<T>(values: &[T], expected: &HashMap<T, usize>) -> CheckerResult
where
    T: Eq + Hash + Debug + Clone,
{
    let observed = build_histogram(values);
    if observed == *expected {
        return CheckerResult::passed();
    }

    let mut issues = Vec::new();
    for (key, expected_count) in expected {
        let actual = observed.get(key).copied().unwrap_or(0);
        if actual != *expected_count {
            issues.push(format!(
                "{:?}: expected {}, found {}",
                key, expected_count, actual
            ));
        }
    }
    for (key, actual) in &observed {
        if !expected.contains_key(key) {
            issues.push(format!("unexpected key {:?} with count {}", key, actual));
        }
    }

    CheckerResult::failed(format!("histogram mismatch: {}", issues.join(", ")))
}

/// Builds a histogram (frequency table) from the provided values.
pub fn build_histogram<T>(values: &[T]) -> HashMap<T, usize>
where
    T: Eq + Hash + Clone,
{
    let mut counts: HashMap<T, usize> = HashMap::with_capacity(values.len());
    for value in values {
        *counts.entry(value.clone()).or_insert(0) += 1;
    }
    counts
}
