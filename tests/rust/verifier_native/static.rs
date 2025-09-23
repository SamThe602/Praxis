use praxis_verifier_native::r#static::{
    analyze, ArrayAccess, ArraySpec, IntRange, LengthRange, LoopBound, LoopSpec, MapAccess,
    MapKeyConstraint, MapSpec, StaticAnalysisInput, StaticViolationKind, ViolationKind,
};

#[test]
fn reports_out_of_bounds_array_access() {
    let mut input = StaticAnalysisInput::new();
    input
        .arrays
        .push(ArraySpec::new("values", LengthRange::new(0, Some(5))));
    input
        .array_accesses
        .push(ArrayAccess::new(0, IntRange::new(0, Some(6))));

    let report = analyze(&input);
    assert!(!report.ok(), "analysis should detect the violation");
    let summary = report.summary();
    assert!(!summary.ok);
    assert_eq!(summary.violation_count, 1);
    assert_eq!(
        summary.first_violation_kind,
        StaticViolationKind::ArrayOutOfBounds
    );
    assert_eq!(summary.first_violation_subject, 0);
    assert_eq!(summary.first_violation_observed, 6);
    assert_eq!(summary.first_violation_expected, 4);
    assert!(report.violations[0].message.contains("exceeds upper bound"));
}

#[test]
fn accepts_well_bounded_program() {
    let mut input = StaticAnalysisInput::new();
    input
        .arrays
        .push(ArraySpec::new("values", LengthRange::new(2, Some(6))));
    input
        .maps
        .push(MapSpec::new("counts", ["low", "high"], false));
    input
        .array_accesses
        .push(ArrayAccess::new(0, IntRange::new(0, Some(5))));
    input.map_accesses.push(MapAccess::new(
        0,
        MapKeyConstraint::Known("low".to_string()),
    ));
    input.loops.push(LoopSpec::new(
        "walk_values",
        LoopBound::DerivedFromArray {
            array: 0,
            stride: 1,
            floor: 0,
        },
    ));

    let report = analyze(&input);
    assert!(report.ok(), "analysis should succeed on safe inputs");
    assert!(report.summary().ok);
}

#[test]
fn flags_missing_map_keys_and_unbounded_loops() {
    let mut input = StaticAnalysisInput::new();
    input
        .arrays
        .push(ArraySpec::new("stream", LengthRange::new(0, None)));
    input.maps.push(MapSpec::new("counts", ["seen"], false));
    input.map_accesses.push(MapAccess::new(
        0,
        MapKeyConstraint::Known("missing".to_string()),
    ));
    input.loops.push(LoopSpec::new(
        "scan_stream",
        LoopBound::DerivedFromArray {
            array: 0,
            stride: 1,
            floor: 0,
        },
    ));

    let report = analyze(&input);
    assert_eq!(report.violations.len(), 2, "two violations expected");
    let kinds: Vec<_> = report
        .violations
        .iter()
        .map(|violation| violation.kind)
        .collect();
    assert!(
        kinds.contains(&ViolationKind::MapMissingKey),
        "missing key should be flagged"
    );
    assert!(
        kinds.contains(&ViolationKind::LoopUnbounded),
        "unbounded loop should be flagged"
    );
}
