//! Minimal C-compatible bridge between Python and the Praxis VM runtime.
//!
//! The interface intentionally uses JSON payloads so the Python orchestrator and
//! verifier can evolve independently of the Rust ABI.  Callers pass in a
//! request describing the bytecode module, entry function, arguments, and
//! execution limits; the runtime responds with either a result + trace or an
//! error descriptor.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString};
use std::ptr;
use std::time::Duration;

use libc::c_char;
use ordered_float::OrderedFloat;
use praxis_verifier_native::checkers as native_checkers;
use praxis_verifier_native::metamorphic as native_metamorphic;
use praxis_verifier_native::r#static::{
    self as native_static, analyze as native_static_analyze, ArrayAccess as NativeArrayAccess,
    ArraySpec as NativeArraySpec, IntRange as NativeIntRange, LengthRange as NativeLengthRange,
    LoopBound as NativeLoopBound, LoopSpec as NativeLoopSpec, MapAccess as NativeMapAccess,
    MapKeyConstraint as NativeMapKeyConstraint, MapSpec as NativeMapSpec,
    StaticAnalysisInput as NativeStaticInput, ViolationDetail as NativeViolationDetail,
};
use praxis_vm_runtime::builtins::Builtins;
use praxis_vm_runtime::interpreter::{ExecutionConfig, ExecutionOutcome, Interpreter, VmError};
use praxis_vm_runtime::memory::Value;
use praxis_vm_runtime::sandbox::{SandboxLimits, SandboxMetrics};
use praxis_vm_runtime::trace::ExecutionTrace;
use praxis_vm_runtime::{bytecode::BytecodeModule, memory::MemoryMetrics};

use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;
use verifier_native_bridge::{
    dispatch_checker_call, dispatch_metamorphic_call, dispatch_static_analyze,
};

#[derive(Debug, Deserialize)]
struct ExecutionRequest {
    module: BytecodeModule,
    entry: String,
    #[serde(default)]
    args: Vec<Value>,
    #[serde(default)]
    limits: Option<RequestLimits>,
}

#[derive(Debug, Deserialize)]
struct RequestLimits {
    #[serde(default)]
    instruction_limit: Option<u64>,
    #[serde(default)]
    stack_limit: Option<usize>,
    #[serde(default)]
    call_depth_limit: Option<usize>,
    #[serde(default)]
    wall_time_ms: Option<u64>,
    #[serde(default)]
    heap_limit_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ExecutionResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<TraceEnvelope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorPayload>,
}

#[derive(Debug, Serialize)]
struct TraceEnvelope {
    trace: ExecutionTrace,
    sandbox: SandboxMetrics,
    heap: MemoryMetrics,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    kind: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<serde_json::Value>,
}

#[derive(Debug, Error)]
enum BridgeError {
    #[error("received null pointer from caller")]
    NullPointer,
    #[error("ffi payload contained interior null byte")]
    InteriorNull,
    #[error("failed to parse request JSON: {0}")]
    Parse(serde_json::Error),
    #[error("execution failed")] // Details captured separately.
    Execution(VmError),
}

impl From<VmError> for BridgeError {
    fn from(value: VmError) -> Self {
        BridgeError::Execution(value)
    }
}

/// # Safety
///
/// The caller must pass a valid, null-terminated UTF-8 string pointer obtained
/// from foreign code. The returned pointer must be released with
/// [`praxis_vm_free`].
#[no_mangle]
pub unsafe extern "C" fn praxis_vm_execute(request: *const c_char) -> *mut c_char {
    match execute_internal(request) {
        Ok(response) => to_c_string(response),
        Err(error) => to_c_string(error_response(error)),
    }
}

/// # Safety
///
/// The returned pointer owns a heap allocation and must be released with
/// [`praxis_vm_free`].
#[no_mangle]
pub unsafe extern "C" fn praxis_vm_builtins() -> *mut c_char {
    let builtins = Builtins::standard();
    let descriptors = builtins.descriptors();
    match serde_json::to_string(&json!({
        "ok": true,
        "builtins": descriptors,
    })) {
        Ok(serialised) => CString::new(serialised).map_or(ptr::null_mut(), CString::into_raw),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
///
/// `ptr` must originate from [`praxis_vm_execute`] or [`praxis_vm_builtins`].
#[no_mangle]
pub unsafe extern "C" fn praxis_vm_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// # Safety
///
/// Caller must supply a valid JSON payload matching the native static analysis schema.
#[no_mangle]
pub unsafe extern "C" fn praxis_verifier_static_analyze(request: *const c_char) -> *mut c_char {
    dispatch_static_analyze(request)
}

/// # Safety
///
/// Caller must supply a valid JSON payload describing the relation evaluation.
#[no_mangle]
pub unsafe extern "C" fn praxis_verifier_metamorphic(request: *const c_char) -> *mut c_char {
    dispatch_metamorphic_call(request)
}

/// # Safety
///
/// Caller must supply a valid JSON payload describing the checker evaluation.
#[no_mangle]
pub unsafe extern "C" fn praxis_verifier_checker(request: *const c_char) -> *mut c_char {
    dispatch_checker_call(request)
}

unsafe fn execute_internal(request: *const c_char) -> Result<ExecutionResponse, BridgeError> {
    if request.is_null() {
        return Err(BridgeError::NullPointer);
    }
    let c_str = CStr::from_ptr(request);
    let raw = c_str.to_str().map_err(|_| BridgeError::InteriorNull)?;
    let payload: ExecutionRequest = serde_json::from_str(raw).map_err(BridgeError::Parse)?;

    let config = build_config(payload.limits);
    let interpreter = Interpreter::with_config(payload.module, config);
    let outcome = interpreter
        .execute(&payload.entry, payload.args)
        .map_err(BridgeError::Execution)?;

    Ok(success_response(outcome))
}

fn build_config(limits: Option<RequestLimits>) -> ExecutionConfig {
    let mut sandbox_limits = SandboxLimits::default();
    let mut heap_limit_bytes = sandbox_limits.heap_limit_bytes;

    if let Some(limits) = limits {
        if let Some(instruction) = limits.instruction_limit {
            sandbox_limits.instruction_limit = instruction;
        }
        if let Some(stack) = limits.stack_limit {
            sandbox_limits.stack_limit = stack;
        }
        if let Some(call_depth) = limits.call_depth_limit {
            sandbox_limits.call_depth_limit = call_depth;
        }
        if let Some(ms) = limits.wall_time_ms {
            sandbox_limits.wall_time = Duration::from_millis(ms);
        }
        if let Some(heap) = limits.heap_limit_bytes {
            heap_limit_bytes = heap;
            sandbox_limits.heap_limit_bytes = heap;
        }
    }

    ExecutionConfig {
        sandbox_limits,
        heap_limit_bytes,
    }
}

fn success_response(outcome: ExecutionOutcome) -> ExecutionResponse {
    let ExecutionOutcome {
        return_value,
        trace,
    } = outcome;
    let sandbox_metrics = trace.metrics.sandbox;
    let heap_metrics = trace.metrics.heap.clone();
    ExecutionResponse {
        ok: true,
        value: return_value,
        trace: Some(TraceEnvelope {
            sandbox: sandbox_metrics,
            heap: heap_metrics,
            trace,
        }),
        error: None,
    }
}

fn error_response(error: BridgeError) -> ExecutionResponse {
    let payload = match error {
        BridgeError::NullPointer => ErrorPayload {
            kind: "null_pointer".to_string(),
            message: "received null pointer".to_string(),
            detail: None,
        },
        BridgeError::InteriorNull => ErrorPayload {
            kind: "invalid_utf8".to_string(),
            message: "payload contained interior null byte".to_string(),
            detail: None,
        },
        BridgeError::Parse(err) => ErrorPayload {
            kind: "parse_error".to_string(),
            message: err.to_string(),
            detail: None,
        },
        BridgeError::Execution(vm_error) => classify_vm_error(vm_error),
    };
    ExecutionResponse {
        ok: false,
        value: None,
        trace: None,
        error: Some(payload),
    }
}

fn classify_vm_error(error: VmError) -> ErrorPayload {
    match error {
        VmError::FunctionNotFound(name) => ErrorPayload {
            kind: "function_not_found".to_string(),
            message: format!("function '{name}' was not found in module"),
            detail: None,
        },
        VmError::FunctionHandleOutOfBounds(index) => ErrorPayload {
            kind: "invalid_function_handle".to_string(),
            message: format!("function handle index {index} out of bounds"),
            detail: None,
        },
        VmError::ConstantOutOfBounds(index) => ErrorPayload {
            kind: "constant_out_of_bounds".to_string(),
            message: format!("constant index {index} is out of range"),
            detail: None,
        },
        VmError::InvalidRegister(index) => ErrorPayload {
            kind: "invalid_register".to_string(),
            message: format!("register index {index} is invalid for this frame"),
            detail: None,
        },
        VmError::StackUnderflow(op) => ErrorPayload {
            kind: "stack_underflow".to_string(),
            message: format!("stack underflow during {op}"),
            detail: None,
        },
        VmError::TypeMismatch { expected, found } => ErrorPayload {
            kind: "type_mismatch".to_string(),
            message: format!("expected {expected} but found {found}"),
            detail: None,
        },
        VmError::InvalidOperands(opcode) => ErrorPayload {
            kind: "invalid_operands".to_string(),
            message: format!("opcode {opcode} received invalid operands"),
            detail: None,
        },
        VmError::InvalidState(message) => ErrorPayload {
            kind: "invalid_state".to_string(),
            message: message.to_string(),
            detail: None,
        },
        VmError::Unsupported(feature) => ErrorPayload {
            kind: "unsupported".to_string(),
            message: feature.to_string(),
            detail: None,
        },
        VmError::ArityMismatch {
            function,
            expected,
            actual,
        } => ErrorPayload {
            kind: "arity_mismatch".to_string(),
            message: format!(
                "function '{function}' expected {expected} arguments but received {actual}"
            ),
            detail: None,
        },
        VmError::Builtin(err) => ErrorPayload {
            kind: "builtin_error".to_string(),
            message: err.to_string(),
            detail: None,
        },
        VmError::Memory(err) => ErrorPayload {
            kind: "memory_error".to_string(),
            message: err.to_string(),
            detail: None,
        },
        VmError::Sandbox(err) => ErrorPayload {
            kind: "sandbox_error".to_string(),
            message: err.to_string(),
            detail: None,
        },
        VmError::ContractViolated(binding) => ErrorPayload {
            kind: "contract_violation".to_string(),
            message: "contract assertion failed".to_string(),
            detail: Some(
                json!({ "resource": binding.binding.resource, "detail": binding.binding.detail }),
            ),
        },
    }
}

fn to_c_string(response: ExecutionResponse) -> *mut c_char {
    match serde_json::to_string(&response) {
        Ok(serialised) => CString::new(serialised)
            .map(CString::into_raw)
            .unwrap_or(ptr::null_mut()),
        Err(_) => ptr::null_mut(),
    }
}

mod verifier_native_bridge {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, thiserror::Error)]
    enum NativeBridgeError {
        #[error("received null pointer from caller")]
        NullPointer,
        #[error("ffi payload contained interior null byte")]
        InteriorNull,
        #[error("failed to parse request JSON: {0}")]
        Parse(serde_json::Error),
        #[error("invalid request: {0}")]
        Invalid(String),
        #[error("serialization failure: {0}")]
        Serialize(serde_json::Error),
    }

    pub(super) unsafe fn dispatch_static_analyze(ptr: *const c_char) -> *mut c_char {
        match static_analyze(ptr) {
            Ok(payload) => json_to_c_string(&payload),
            Err(error) => json_to_c_string(&error_payload(error)),
        }
    }

    pub(super) unsafe fn dispatch_metamorphic_call(ptr: *const c_char) -> *mut c_char {
        match metamorphic_call(ptr) {
            Ok(payload) => json_to_c_string(&payload),
            Err(error) => json_to_c_string(&error_payload(error)),
        }
    }

    pub(super) unsafe fn dispatch_checker_call(ptr: *const c_char) -> *mut c_char {
        match checker_call(ptr) {
            Ok(payload) => json_to_c_string(&payload),
            Err(error) => json_to_c_string(&error_payload(error)),
        }
    }

    fn json_to_c_string<T: Serialize>(value: &T) -> *mut c_char {
        match serde_json::to_string(value)
            .map_err(NativeBridgeError::Serialize)
            .and_then(|serialised| {
                CString::new(serialised).map_err(|_| NativeBridgeError::InteriorNull)
            }) {
            Ok(cstring) => cstring.into_raw(),
            Err(error) => {
                let fallback = error_payload(error);
                serde_json::to_string(&fallback)
                    .ok()
                    .and_then(|text| CString::new(text).ok())
                    .map_or(ptr::null_mut(), CString::into_raw)
            }
        }
    }

    fn error_payload(error: NativeBridgeError) -> serde_json::Value {
        serde_json::json!({
            "ok": false,
            "error": {
                "kind": "native_bridge_error",
                "message": error.to_string(),
            }
        })
    }

    unsafe fn read_request(ptr: *const c_char) -> Result<&'static str, NativeBridgeError> {
        if ptr.is_null() {
            return Err(NativeBridgeError::NullPointer);
        }
        let c_str = CStr::from_ptr(ptr);
        c_str.to_str().map_err(|_| NativeBridgeError::InteriorNull)
    }

    unsafe fn parse_request<T>(ptr: *const c_char) -> Result<T, NativeBridgeError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let raw = read_request(ptr)?;
        serde_json::from_str(raw).map_err(NativeBridgeError::Parse)
    }

    #[derive(Debug, Deserialize)]
    struct StaticAnalysisRequest {
        #[serde(default)]
        arrays: Vec<StaticArraySpec>,
        #[serde(default)]
        maps: Vec<StaticMapSpec>,
        #[serde(default)]
        loops: Vec<StaticLoopSpec>,
        #[serde(default)]
        array_accesses: Vec<StaticArrayAccess>,
        #[serde(default)]
        map_accesses: Vec<StaticMapAccess>,
    }

    #[derive(Debug, Deserialize)]
    struct StaticArraySpec {
        name: String,
        length: StaticLengthRange,
    }

    #[derive(Debug, Deserialize)]
    struct StaticLengthRange {
        min: usize,
        #[serde(default)]
        max: Option<usize>,
    }

    #[derive(Debug, Deserialize)]
    struct StaticMapSpec {
        name: String,
        #[serde(default)]
        known_keys: Vec<String>,
        #[serde(default)]
        allow_unknown_keys: bool,
    }

    #[derive(Debug, Deserialize)]
    struct StaticLoopSpec {
        name: String,
        bound: LoopBoundRequest,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum LoopBoundRequest {
        Explicit {
            min: usize,
            #[serde(default)]
            max: Option<usize>,
        },
        DerivedFromArray {
            array: usize,
            stride: usize,
            #[serde(default)]
            floor: usize,
        },
    }

    #[derive(Debug, Deserialize)]
    struct StaticArrayAccess {
        array: usize,
        index: StaticIntRange,
    }

    #[derive(Debug, Deserialize)]
    struct StaticIntRange {
        #[serde(default)]
        min: isize,
        #[serde(default)]
        max: Option<isize>,
    }

    #[derive(Debug, Deserialize)]
    struct StaticMapAccess {
        map: usize,
        key: MapKeyRequest,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum MapKeyRequest {
        Known { value: String },
        OneOf { values: Vec<String> },
        Unknown,
    }

    #[derive(Debug, Serialize)]
    struct StaticAnalysisResponse {
        ok: bool,
        violation_count: usize,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        violations: Vec<StaticViolationPayload>,
    }

    #[derive(Debug, Serialize)]
    struct StaticViolationPayload {
        kind: String,
        subject: u32,
        observed: i64,
        expected: i64,
        message: String,
    }

    fn static_analyze(ptr: *const c_char) -> Result<StaticAnalysisResponse, NativeBridgeError> {
        unsafe {
            let request: StaticAnalysisRequest = parse_request(ptr)?;
            let input = request.into_native()?;
            let report = native_static_analyze(&input);
            let violations = report
                .violations
                .iter()
                .map(StaticViolationPayload::from)
                .collect::<Vec<_>>();
            Ok(StaticAnalysisResponse {
                ok: report.ok(),
                violation_count: violations.len(),
                violations,
            })
        }
    }

    impl StaticAnalysisRequest {
        fn into_native(self) -> Result<NativeStaticInput, NativeBridgeError> {
            let mut input = NativeStaticInput::new();
            input.arrays = self
                .arrays
                .into_iter()
                .map(|spec| spec.into_native())
                .collect::<Result<Vec<_>, _>>()?;
            input.maps = self
                .maps
                .into_iter()
                .map(|spec| spec.into_native())
                .collect::<Result<Vec<_>, _>>()?;
            input.loops = self
                .loops
                .into_iter()
                .map(|spec| spec.into_native())
                .collect::<Result<Vec<_>, _>>()?;
            input.array_accesses = self
                .array_accesses
                .into_iter()
                .map(|access| Ok(access.into_native()))
                .collect::<Result<Vec<_>, _>>()?;
            input.map_accesses = self
                .map_accesses
                .into_iter()
                .map(|access| access.into_native())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(input)
        }
    }

    impl StaticArraySpec {
        fn into_native(self) -> Result<NativeArraySpec, NativeBridgeError> {
            Ok(NativeArraySpec::new(self.name, self.length.into_native()))
        }
    }

    impl StaticLengthRange {
        fn into_native(self) -> NativeLengthRange {
            NativeLengthRange::new(self.min, self.max)
        }
    }

    impl StaticMapSpec {
        fn into_native(self) -> Result<NativeMapSpec, NativeBridgeError> {
            Ok(NativeMapSpec::new(
                self.name,
                self.known_keys,
                self.allow_unknown_keys,
            ))
        }
    }

    impl StaticLoopSpec {
        fn into_native(self) -> Result<NativeLoopSpec, NativeBridgeError> {
            Ok(NativeLoopSpec::new(self.name, self.bound.into_native()?))
        }
    }

    impl LoopBoundRequest {
        fn into_native(self) -> Result<NativeLoopBound, NativeBridgeError> {
            match self {
                LoopBoundRequest::Explicit { min, max } => {
                    Ok(NativeLoopBound::Explicit { min, max })
                }
                LoopBoundRequest::DerivedFromArray {
                    array,
                    stride,
                    floor,
                } => {
                    if stride == 0 {
                        return Err(NativeBridgeError::Invalid(
                            "loop stride must be greater than zero".to_string(),
                        ));
                    }
                    Ok(NativeLoopBound::DerivedFromArray {
                        array,
                        stride,
                        floor,
                    })
                }
            }
        }
    }

    impl StaticArrayAccess {
        fn into_native(self) -> NativeArrayAccess {
            NativeArrayAccess::new(self.array, self.index.into_native())
        }
    }

    impl StaticIntRange {
        fn into_native(self) -> NativeIntRange {
            NativeIntRange::new(self.min, self.max)
        }
    }

    impl StaticMapAccess {
        fn into_native(self) -> Result<NativeMapAccess, NativeBridgeError> {
            Ok(NativeMapAccess::new(self.map, self.key.into_native()?))
        }
    }

    impl MapKeyRequest {
        fn into_native(self) -> Result<NativeMapKeyConstraint, NativeBridgeError> {
            match self {
                MapKeyRequest::Known { value } => Ok(NativeMapKeyConstraint::Known(value)),
                MapKeyRequest::OneOf { values } => Ok(NativeMapKeyConstraint::OneOf(values)),
                MapKeyRequest::Unknown => Ok(NativeMapKeyConstraint::Unknown),
            }
        }
    }

    impl From<&NativeViolationDetail> for StaticViolationPayload {
        fn from(detail: &NativeViolationDetail) -> Self {
            Self {
                kind: match detail.kind {
                    native_static::ViolationKind::ArrayOutOfBounds => "array_out_of_bounds",
                    native_static::ViolationKind::MapMissingKey => "map_missing_key",
                    native_static::ViolationKind::LoopUnbounded => "loop_unbounded",
                }
                .to_string(),
                subject: detail.subject,
                observed: detail.observed,
                expected: detail.expected,
                message: detail.message.clone(),
            }
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "relation", rename_all = "snake_case")]
    enum MetamorphicRequest {
        PermutationInvariance(PermutationPayload),
        Idempotence(IdempotencePayload),
        InverseProperty(InversePayload),
        Monotonicity(MonotonicityPayload),
    }

    #[derive(Debug, Deserialize, Clone)]
    struct PermutationPayload {
        base_inputs: Vec<serde_json::Value>,
        base_output: serde_json::Value,
        #[serde(default)]
        variants: Vec<PermutationVariant>,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct PermutationVariant {
        inputs: Vec<serde_json::Value>,
        output: serde_json::Value,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct IdempotencePayload {
        value: serde_json::Value,
        once: serde_json::Value,
        twice: serde_json::Value,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct InversePayload {
        samples: Vec<InverseSample>,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct InverseSample {
        original: serde_json::Value,
        encoded: serde_json::Value,
        decoded: serde_json::Value,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct MonotonicityPayload {
        values: Vec<serde_json::Value>,
        #[serde(default)]
        strict: bool,
    }

    #[derive(Debug, Serialize)]
    struct RelationResponse {
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    fn metamorphic_call(ptr: *const c_char) -> Result<RelationResponse, NativeBridgeError> {
        unsafe {
            let request: MetamorphicRequest = parse_request(ptr)?;
            let result = match request {
                MetamorphicRequest::PermutationInvariance(payload) => evaluate_permutation(payload),
                MetamorphicRequest::Idempotence(payload) => evaluate_idempotence(payload),
                MetamorphicRequest::InverseProperty(payload) => evaluate_inverse(payload),
                MetamorphicRequest::Monotonicity(payload) => evaluate_monotonicity(payload),
            }?;
            Ok(RelationResponse {
                ok: result.ok,
                message: result.message,
            })
        }
    }

    fn evaluate_permutation(
        payload: PermutationPayload,
    ) -> Result<native_metamorphic::RelationResult, NativeBridgeError> {
        let PermutationPayload {
            base_inputs,
            base_output,
            variants,
        } = payload;
        let mut outputs = HashMap::new();
        let base_key = serde_json::to_string(&base_inputs).map_err(NativeBridgeError::Serialize)?;
        outputs.insert(base_key, base_output);
        for variant in &variants {
            let key =
                serde_json::to_string(&variant.inputs).map_err(NativeBridgeError::Serialize)?;
            outputs.insert(key, variant.output.clone());
        }
        let variant_inputs = variants
            .into_iter()
            .map(|variant| variant.inputs)
            .collect::<Vec<_>>();
        Ok(native_metamorphic::permutation_invariance(
            &base_inputs,
            variant_inputs,
            |inputs| {
                let key = serde_json::to_string(inputs).unwrap_or_default();
                outputs
                    .get(&key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null)
            },
        ))
    }

    fn evaluate_idempotence(
        payload: IdempotencePayload,
    ) -> Result<native_metamorphic::RelationResult, NativeBridgeError> {
        let outputs = RefCell::new(vec![payload.once, payload.twice]);
        Ok(native_metamorphic::idempotence(payload.value, |value| {
            let mut guard = outputs.borrow_mut();
            if guard.is_empty() {
                value
            } else {
                guard.remove(0)
            }
        }))
    }

    fn evaluate_inverse(
        payload: InversePayload,
    ) -> Result<native_metamorphic::RelationResult, NativeBridgeError> {
        let mut forward = HashMap::new();
        let mut inverse = HashMap::new();
        let mut samples = Vec::new();
        for sample in payload.samples {
            let original_key =
                serde_json::to_string(&sample.original).map_err(NativeBridgeError::Serialize)?;
            let encoded_key =
                serde_json::to_string(&sample.encoded).map_err(NativeBridgeError::Serialize)?;
            forward.insert(original_key, sample.encoded.clone());
            inverse.insert(encoded_key, sample.decoded.clone());
            samples.push(sample.original);
        }
        Ok(native_metamorphic::inverse_property(
            &samples,
            |value| {
                let key = serde_json::to_string(&value).unwrap_or_default();
                forward
                    .get(&key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null)
            },
            |value| {
                let key = serde_json::to_string(&value).unwrap_or_default();
                inverse
                    .get(&key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null)
            },
        ))
    }

    fn evaluate_monotonicity(
        payload: MonotonicityPayload,
    ) -> Result<native_metamorphic::RelationResult, NativeBridgeError> {
        let MonotonicityPayload { values, strict } = payload;
        let outputs = coerce_values(values)?;
        let inputs: Vec<usize> = (0..outputs.len()).collect();
        Ok(native_metamorphic::monotonicity(
            &inputs,
            |index: &usize| {
                outputs
                    .get(*index)
                    .cloned()
                    .unwrap_or_else(|| outputs.last().cloned().unwrap_or(SimpleValue::Integer(0)))
            },
            strict,
        ))
    }

    #[derive(Debug, Deserialize)]
    struct CheckerRequest {
        checker: String,
        #[serde(default)]
        values: Vec<serde_json::Value>,
        #[serde(default)]
        expected: Option<BTreeMap<String, usize>>,
    }

    #[derive(Debug, Serialize)]
    struct CheckerResponse {
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    fn checker_call(ptr: *const c_char) -> Result<CheckerResponse, NativeBridgeError> {
        unsafe {
            let request: CheckerRequest = parse_request(ptr)?;
            let result = match request.checker.as_str() {
                "sorted" => evaluate_sorted(request.values)?,
                "unique" => evaluate_unique(request.values)?,
                "histogram_matches" => evaluate_histogram(request.values, request.expected)?,
                other => {
                    return Err(NativeBridgeError::Invalid(format!(
                        "unknown checker '{other}'"
                    )))
                }
            };
            Ok(CheckerResponse {
                ok: result.ok,
                message: result.message,
            })
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum SimpleValue {
        Integer(i64),
        Float(OrderedFloat<f64>),
        String(String),
        Bool(bool),
    }

    impl PartialOrd for SimpleValue {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for SimpleValue {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            use SimpleValue::*;
            match (self, other) {
                (Integer(lhs), Integer(rhs)) => lhs.cmp(rhs),
                (Float(lhs), Float(rhs)) => lhs.cmp(rhs),
                (String(lhs), String(rhs)) => lhs.cmp(rhs),
                (Bool(lhs), Bool(rhs)) => lhs.cmp(rhs),
                (Integer(_), Float(_)) => std::cmp::Ordering::Less,
                (Float(_), Integer(_)) => std::cmp::Ordering::Greater,
                (Integer(_), String(_)) | (Float(_), String(_)) => std::cmp::Ordering::Less,
                (String(_), Integer(_)) | (String(_), Float(_)) => std::cmp::Ordering::Greater,
                (Bool(_), _) => std::cmp::Ordering::Greater,
                (_, Bool(_)) => std::cmp::Ordering::Less,
            }
        }
    }

    fn coerce_values(
        values: Vec<serde_json::Value>,
    ) -> Result<Vec<SimpleValue>, NativeBridgeError> {
        values.into_iter().map(SimpleValue::try_from).collect()
    }

    impl TryFrom<serde_json::Value> for SimpleValue {
        type Error = NativeBridgeError;

        fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
            match value {
                serde_json::Value::Number(number) => {
                    if let Some(int) = number.as_i64() {
                        Ok(SimpleValue::Integer(int))
                    } else if let Some(unsigned) = number.as_u64() {
                        Ok(SimpleValue::Integer(unsigned as i64))
                    } else if let Some(float) = number.as_f64() {
                        Ok(SimpleValue::Float(OrderedFloat(float)))
                    } else {
                        Err(NativeBridgeError::Invalid(
                            "unsupported numeric value".to_string(),
                        ))
                    }
                }
                serde_json::Value::String(text) => Ok(SimpleValue::String(text)),
                serde_json::Value::Bool(flag) => Ok(SimpleValue::Bool(flag)),
                other => Err(NativeBridgeError::Invalid(format!(
                    "unsupported value type: {}",
                    other
                ))),
            }
        }
    }

    fn evaluate_sorted(
        values: Vec<serde_json::Value>,
    ) -> Result<native_checkers::CheckerResult, NativeBridgeError> {
        let coerced = coerce_values(values)?;
        Ok(native_checkers::sorted(&coerced))
    }

    fn evaluate_unique(
        values: Vec<serde_json::Value>,
    ) -> Result<native_checkers::CheckerResult, NativeBridgeError> {
        let coerced = coerce_values(values)?;
        Ok(native_checkers::unique(&coerced))
    }

    fn evaluate_histogram(
        values: Vec<serde_json::Value>,
        expected: Option<BTreeMap<String, usize>>,
    ) -> Result<native_checkers::CheckerResult, NativeBridgeError> {
        let expected = expected.ok_or_else(|| {
            NativeBridgeError::Invalid("histogram_matches requires 'expected' payload".to_string())
        })?;
        let coerced = coerce_values(values)?;
        let mut converted: HashMap<SimpleValue, usize> = HashMap::new();
        for (key, count) in expected {
            let json = serde_json::from_str::<serde_json::Value>(&key)
                .unwrap_or(serde_json::Value::String(key));
            let value = SimpleValue::try_from(json)?;
            converted.insert(value, count);
        }
        Ok(native_checkers::histogram_matches(&coerced, &converted))
    }
}
