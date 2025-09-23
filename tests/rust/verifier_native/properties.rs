use std::collections::HashMap;

use praxis_verifier_native::checkers::{build_histogram, histogram_matches, sorted, unique};
use praxis_verifier_native::metamorphic::{
    idempotence, inverse_property, monotonicity, permutation_invariance,
};

#[test]
fn permutation_invariance_detects_differences() {
    let base = vec![1, 2, 3];
    let result = permutation_invariance(&base, vec![vec![2, 1, 3], vec![3, 1, 2]], |values| {
        values.iter().sum::<i32>()
    });
    assert!(
        result.ok,
        "sum should be permutation invariant: {:?}",
        result.message
    );

    let failing = permutation_invariance(&base, vec![vec![3, 1, 2]], |values| values[0]);
    assert!(
        !failing.ok,
        "dependent on order should fail: {:?}",
        failing.message
    );
}

#[test]
fn idempotence_and_inverse_relations() {
    let idempotent = idempotence(vec![3, 1, 2], |mut values| {
        values.sort();
        values
    });
    assert!(
        idempotent.ok,
        "sorting should be idempotent: {:?}",
        idempotent.message
    );

    let non_idempotent = idempotence(vec![1, 2], |mut values| {
        values.push(99);
        values
    });
    assert!(!non_idempotent.ok, "push should break idempotence");

    let inverse_ok = inverse_property(&[1, 2, 3], |value| value + 5, |value| value - 5);
    assert!(inverse_ok.ok, "addition/subtraction should be inverse");

    let inverse_fail = inverse_property(&[2, 4], |value| value * 2, |value| value / 2 + 1);
    assert!(!inverse_fail.ok, "incorrect inverse must fail");
}

#[test]
fn monotonicity_checks_cover_strict_and_relaxed_modes() {
    let strict_ok = monotonicity(&[1, 2, 3, 4], |&value| value * 2, true);
    assert!(strict_ok.ok);

    let relaxed_ok = monotonicity(&[1, 2, 2, 3], |&value| value, false);
    assert!(relaxed_ok.ok);

    let failing = monotonicity(&[1, 2, 3], |&value| -value, false);
    assert!(!failing.ok, "decreasing result must violate monotonicity");
}

#[test]
fn sorted_and_unique_checkers() {
    let sorted_ok = sorted(&[1, 2, 3, 4]);
    assert!(sorted_ok.ok);

    let sorted_fail = sorted(&[2, 1, 3]);
    assert!(!sorted_fail.ok);
    assert!(sorted_fail
        .message
        .as_deref()
        .unwrap()
        .contains("not sorted"));

    let unique_ok = unique(&['a', 'b', 'c']);
    assert!(unique_ok.ok);

    let unique_fail = unique(&['x', 'y', 'x']);
    assert!(!unique_fail.ok);
    assert!(unique_fail
        .message
        .as_deref()
        .unwrap()
        .contains("duplicate"));
}

#[test]
fn histogram_helpers_validate_counts() {
    let mut expected = HashMap::new();
    expected.insert('a', 2usize);
    expected.insert('b', 1usize);

    let ok = histogram_matches(&['a', 'b', 'a'], &expected);
    assert!(ok.ok, "exact histogram should pass: {:?}", ok.message);

    let built = build_histogram(&['a', 'b', 'a']);
    assert_eq!(built, expected, "builder must mirror expected counts");

    let mut mismatch_expected = expected.clone();
    mismatch_expected.insert('c', 1);
    let mismatch = histogram_matches(&['a', 'b', 'a'], &mismatch_expected);
    assert!(!mismatch.ok);
    assert!(mismatch
        .message
        .as_deref()
        .unwrap()
        .contains("histogram mismatch"));
}
