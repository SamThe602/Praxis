//! Metamorphic testing combinators implemented in Rust for speed-critical cases.
//!
//! The helpers here intentionally keep their API surface small: they return a
//! [`RelationResult`] describing whether the invariant holds and an optional
//! diagnostic message.  The Python verifier can consume these results directly
//! or through FFI without needing to ship detailed traces across the boundary.

use std::fmt::Debug;

/// Lightweight status for a metamorphic relation evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationResult {
    pub ok: bool,
    pub message: Option<String>,
}

impl RelationResult {
    /// Successful evaluation.
    pub const fn passed() -> Self {
        Self {
            ok: true,
            message: None,
        }
    }

    /// Failed evaluation with a human friendly message.
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: Some(message.into()),
        }
    }
}

/// Checks that reordering the inputs does not affect the evaluator's output.
pub fn permutation_invariance<T, U, F, Iter>(
    base_inputs: &[T],
    permutations: Iter,
    mut evaluator: F,
) -> RelationResult
where
    T: Clone,
    U: PartialEq + Debug,
    F: FnMut(&[T]) -> U,
    Iter: IntoIterator<Item = Vec<T>>,
{
    let expected = evaluator(base_inputs);
    for (index, permutation) in permutations.into_iter().enumerate() {
        let actual = evaluator(&permutation);
        if actual != expected {
            return RelationResult::failed(format!(
                "permutation {} changed output: expected {:?}, received {:?}",
                index, expected, actual
            ));
        }
    }
    RelationResult::passed()
}

/// Checks that applying the evaluator twice is equivalent to a single
/// application (idempotence).
pub fn idempotence<T, F>(value: T, mut evaluator: F) -> RelationResult
where
    T: Clone + PartialEq + Debug,
    F: FnMut(T) -> T,
{
    let once = evaluator(value.clone());
    let twice = evaluator(once.clone());
    if once == twice {
        RelationResult::passed()
    } else {
        RelationResult::failed(format!(
            "idempotence violated: first {:?}, second {:?}",
            once, twice
        ))
    }
}

/// Validates that `inverse` undoes the effect of `forward` across the supplied
/// sample entries.
pub fn inverse_property<T, U, F, G>(samples: &[T], mut forward: F, mut inverse: G) -> RelationResult
where
    T: Clone + PartialEq + Debug,
    U: Debug,
    F: FnMut(T) -> U,
    G: FnMut(U) -> T,
{
    for (index, sample) in samples.iter().cloned().enumerate() {
        let encoded = forward(sample.clone());
        let decoded = inverse(encoded);
        if decoded != sample {
            return RelationResult::failed(format!(
                "inverse property failed at sample {}: decoded {:?} != original {:?}",
                index, decoded, sample
            ));
        }
    }
    RelationResult::passed()
}

/// Asserts that the evaluator produces a (strictly) monotonic sequence given
/// ascending inputs.
pub fn monotonicity<T, U, F>(inputs: &[T], mut evaluator: F, strict: bool) -> RelationResult
where
    T: Clone,
    U: PartialOrd + Debug,
    F: FnMut(&T) -> U,
{
    if inputs.len() <= 1 {
        return RelationResult::passed();
    }

    let mut previous = evaluator(&inputs[0]);
    for (index, item) in inputs.iter().enumerate().skip(1) {
        let current = evaluator(item);
        let comparison_ok = if strict {
            current > previous
        } else {
            current >= previous
        };
        if !comparison_ok {
            return RelationResult::failed(format!(
                "monotonicity violated between indices {} and {}: {:?} then {:?}",
                index - 1,
                index,
                previous,
                current
            ));
        }
        previous = current;
    }
    RelationResult::passed()
}
