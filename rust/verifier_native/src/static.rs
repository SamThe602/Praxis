//! Native static analysis helpers providing light-weight bounds checks.
//!
//! The goal of this module is to offer a fast, allocation-friendly analysis
//! that mirrors the high level Python analyser but can be called through FFI
//! without serialising large payloads.  Callers provide coarse container and
//! loop descriptors and we flag obvious violations such as out-of-bounds array
//! accesses, map key mismatches, or unbounded loops.

use std::collections::{BTreeSet, HashMap};

/// Integer interval used when reasoning about indices or iteration counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntRange {
    /// Smallest possible value. Negative values indicate a potential underflow.
    pub min: isize,
    /// Largest possible value. ``None`` means the upper bound is unknown.
    pub max: Option<isize>,
}

impl IntRange {
    /// Creates an exact range representing a single value.
    pub const fn exact(value: isize) -> Self {
        Self {
            min: value,
            max: Some(value),
        }
    }

    /// Creates a range with the provided bounds.
    pub const fn new(min: isize, max: Option<isize>) -> Self {
        Self { min, max }
    }
}

/// Closed interval describing the cardinality of a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthRange {
    pub min: usize,
    pub max: Option<usize>,
}

impl LengthRange {
    /// Constructs a new range, normalising impossible bounds.
    pub fn new(min: usize, max: Option<usize>) -> Self {
        let normalised_max = max.filter(|bound| *bound >= min);
        Self {
            min,
            max: normalised_max,
        }
    }

    /// Returns the greatest valid index based on the maximum length.
    fn max_index(self) -> Option<isize> {
        self.max.map(|len| len.saturating_sub(1) as isize)
    }
}

/// Description of an array/list like container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArraySpec {
    pub name: String,
    pub length: LengthRange,
}

impl ArraySpec {
    pub fn new(name: impl Into<String>, length: LengthRange) -> Self {
        Self {
            name: name.into(),
            length,
        }
    }
}

/// Access to an array using a possibly ranged index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayAccess {
    pub array: usize,
    pub index: IntRange,
}

impl ArrayAccess {
    pub const fn new(array: usize, index: IntRange) -> Self {
        Self { array, index }
    }
}

/// Description of a map/dictionary container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapSpec {
    pub name: String,
    pub known_keys: BTreeSet<String>,
    pub allow_unknown_keys: bool,
}

impl MapSpec {
    pub fn new(
        name: impl Into<String>,
        known_keys: impl IntoIterator<Item = impl Into<String>>,
        allow_unknown_keys: bool,
    ) -> Self {
        let keys = known_keys
            .into_iter()
            .map(|key| key.into())
            .collect::<BTreeSet<_>>();
        Self {
            name: name.into(),
            known_keys: keys,
            allow_unknown_keys,
        }
    }

    fn has_key(&self, key: &str) -> bool {
        self.known_keys.contains(key) || self.allow_unknown_keys
    }
}

/// Specification for keys used in a map access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapKeyConstraint {
    Known(String),
    OneOf(Vec<String>),
    Unknown,
}

impl MapKeyConstraint {
    fn display(&self) -> String {
        match self {
            MapKeyConstraint::Known(key) => key.clone(),
            MapKeyConstraint::OneOf(keys) => keys.join("|"),
            MapKeyConstraint::Unknown => "<unknown>".to_string(),
        }
    }
}

/// Access pattern for a map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapAccess {
    pub map: usize,
    pub key: MapKeyConstraint,
}

impl MapAccess {
    pub fn new(map: usize, key: MapKeyConstraint) -> Self {
        Self { map, key }
    }
}

/// Loop bound descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopBound {
    /// Explicit iteration interval.
    Explicit { min: usize, max: Option<usize> },
    /// Loop derived from traversing an array with the specified stride.
    DerivedFromArray {
        array: usize,
        stride: usize,
        floor: usize,
    },
}

impl LoopBound {
    fn estimate(
        &self,
        arrays: &[ArraySpec],
    ) -> (usize, Option<usize>, Option<String>, Option<usize>) {
        match *self {
            LoopBound::Explicit { min, max } => (min, max, None, None),
            LoopBound::DerivedFromArray {
                array,
                stride,
                floor,
            } => {
                let name = arrays.get(array).map(|spec| spec.name.clone());
                if stride == 0 {
                    return (floor, None, name, Some(array));
                }
                match arrays.get(array).map(|spec| spec.length) {
                    Some(length) => match length.max {
                        Some(max_len) => {
                            let max_iter = max_len.div_ceil(stride);
                            let min_iter = floor.saturating_add(length.min.div_ceil(stride));
                            (min_iter, Some(max_iter), name, Some(array))
                        }
                        None => (floor, None, name, Some(array)),
                    },
                    None => (floor, None, None, Some(array)),
                }
            }
        }
    }
}

/// Loop specification wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSpec {
    pub name: String,
    pub bound: LoopBound,
}

impl LoopSpec {
    pub fn new(name: impl Into<String>, bound: LoopBound) -> Self {
        Self {
            name: name.into(),
            bound,
        }
    }
}

/// Collection of descriptors passed to the native analyser.
#[derive(Debug, Clone, Default)]
pub struct StaticAnalysisInput {
    pub arrays: Vec<ArraySpec>,
    pub maps: Vec<MapSpec>,
    pub loops: Vec<LoopSpec>,
    pub array_accesses: Vec<ArrayAccess>,
    pub map_accesses: Vec<MapAccess>,
}

impl StaticAnalysisInput {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Dialect-neutral violation classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    ArrayOutOfBounds,
    MapMissingKey,
    LoopUnbounded,
}

/// Detailed violation captured during analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViolationDetail {
    pub kind: ViolationKind,
    pub subject: u32,
    pub observed: i64,
    pub expected: i64,
    pub message: String,
}

impl ViolationDetail {
    fn new(
        kind: ViolationKind,
        subject: u32,
        observed: i64,
        expected: i64,
        message: String,
    ) -> Self {
        Self {
            kind,
            subject,
            observed,
            expected,
            message,
        }
    }
}

/// Aggregate report produced by the analyser.
#[derive(Debug, Clone, Default)]
pub struct AnalysisReport {
    pub violations: Vec<ViolationDetail>,
}

impl AnalysisReport {
    /// Returns true when no violations were recorded.
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }

    /// Produces a compact FFI-compatible summary.
    pub fn summary(&self) -> StaticAnalysisResult {
        if let Some(first) = self.violations.first() {
            StaticAnalysisResult {
                ok: false,
                violation_count: self.violations.len() as u32,
                first_violation_kind: first.kind.into(),
                first_violation_subject: first.subject,
                first_violation_observed: first.observed,
                first_violation_expected: first.expected,
            }
        } else {
            StaticAnalysisResult {
                ok: true,
                violation_count: 0,
                ..Default::default()
            }
        }
    }
}

/// Public entry point mirroring the Python analyser's fast checks.
pub fn analyze(input: &StaticAnalysisInput) -> AnalysisReport {
    let mut violations = Vec::new();

    for access in &input.array_accesses {
        if let Some(array) = input.arrays.get(access.array) {
            if access.index.min < 0 {
                let message = format!(
                    "array '{}' may be indexed with negative value {}",
                    array.name, access.index.min
                );
                violations.push(ViolationDetail::new(
                    ViolationKind::ArrayOutOfBounds,
                    access.array as u32,
                    access.index.min as i64,
                    0,
                    message,
                ));
                continue;
            }

            if let Some(max_index) = access.index.max {
                if let Some(limit) = array.length.max_index() {
                    if max_index > limit {
                        let message = format!(
                            "array '{}' access exceeds upper bound: max index {} vs limit {}",
                            array.name, max_index, limit
                        );
                        violations.push(ViolationDetail::new(
                            ViolationKind::ArrayOutOfBounds,
                            access.array as u32,
                            max_index as i64,
                            limit as i64,
                            message,
                        ));
                    }
                }
            } else if array.length.max.is_some() {
                let message = format!(
                    "array '{}' access lacks upper bound while length is finite",
                    array.name
                );
                violations.push(ViolationDetail::new(
                    ViolationKind::ArrayOutOfBounds,
                    access.array as u32,
                    -1,
                    array.length.max_index().map_or(-1, |limit| limit as i64),
                    message,
                ));
            }
        } else {
            let message = format!("array index {} is not declared", access.array);
            violations.push(ViolationDetail::new(
                ViolationKind::ArrayOutOfBounds,
                access.array as u32,
                -1,
                -1,
                message,
            ));
        }
    }

    for access in &input.map_accesses {
        let Some(map) = input.maps.get(access.map) else {
            let message = format!("map index {} is not declared", access.map);
            violations.push(ViolationDetail::new(
                ViolationKind::MapMissingKey,
                access.map as u32,
                0,
                1,
                message,
            ));
            continue;
        };

        let satisfied = match &access.key {
            MapKeyConstraint::Known(key) => map.has_key(key),
            MapKeyConstraint::OneOf(keys) => keys.iter().all(|key| map.has_key(key)),
            MapKeyConstraint::Unknown => map.allow_unknown_keys,
        };

        if !satisfied {
            let message = format!(
                "map '{}' missing required key(s): {}",
                map.name,
                access.key.display()
            );
            violations.push(ViolationDetail::new(
                ViolationKind::MapMissingKey,
                access.map as u32,
                0,
                1,
                message,
            ));
        }
    }

    for (loop_index, loop_spec) in input.loops.iter().enumerate() {
        let (min_iter, max_iter, source_array_name, source_index) =
            loop_spec.bound.estimate(&input.arrays);
        let mut expected = max_iter.map_or(-1, |value| value as i64);
        let mut observed = -1;
        let mut violation = false;
        if let Some(maximum) = max_iter {
            if maximum < min_iter {
                violation = true;
                observed = maximum as i64;
                expected = min_iter as i64;
            }
        } else {
            violation = true;
        }

        if violation {
            let name = &loop_spec.name;
            let message = match source_array_name {
                Some(array_name) => format!(
                    "loop '{}' derived from array '{}' lacks a finite upper bound",
                    name, array_name
                ),
                None => format!("loop '{}' lacks a finite upper bound", name),
            };
            let subject = source_index.map_or(loop_index as u32, |idx| idx as u32);
            violations.push(ViolationDetail::new(
                ViolationKind::LoopUnbounded,
                subject,
                observed,
                expected,
                message,
            ));
        }
    }

    AnalysisReport { violations }
}

/// FFI classification for violations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticViolationKind {
    None = 0,
    ArrayOutOfBounds = 1,
    MapMissingKey = 2,
    LoopUnbounded = 3,
}

impl From<ViolationKind> for StaticViolationKind {
    fn from(value: ViolationKind) -> Self {
        match value {
            ViolationKind::ArrayOutOfBounds => StaticViolationKind::ArrayOutOfBounds,
            ViolationKind::MapMissingKey => StaticViolationKind::MapMissingKey,
            ViolationKind::LoopUnbounded => StaticViolationKind::LoopUnbounded,
        }
    }
}

/// Lightweight FFI-facing summary of the analysis.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticAnalysisResult {
    pub ok: bool,
    pub violation_count: u32,
    pub first_violation_kind: StaticViolationKind,
    pub first_violation_subject: u32,
    pub first_violation_observed: i64,
    pub first_violation_expected: i64,
}

impl Default for StaticAnalysisResult {
    fn default() -> Self {
        Self {
            ok: true,
            violation_count: 0,
            first_violation_kind: StaticViolationKind::None,
            first_violation_subject: 0,
            first_violation_observed: 0,
            first_violation_expected: 0,
        }
    }
}

/// Helper converting the report into a lookup map for richer consumers.
pub fn violation_map(report: &AnalysisReport) -> HashMap<u32, Vec<&ViolationDetail>> {
    let mut grouped: HashMap<u32, Vec<&ViolationDetail>> = HashMap::new();
    for violation in &report.violations {
        grouped
            .entry(violation.subject)
            .or_default()
            .push(violation);
    }
    grouped
}
