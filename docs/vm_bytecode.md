# VM Bytecode

The Praxis VM executes a compact bytecode format that mirrors the mid-level IR
produced by the compiler.  This document summarises the execution contract,
trace schema, and sandbox controls surfaced through the Rust runtime and Python
FFI.

## Execution contract

The native FFI accepts and returns UTF-8 JSON payloads.  The Python runner in
`packages.verifier.runner` assembles the request shown below:

```json
{
  "module": {"constants": [...], "functions": [...]},
  "entry": "solve",
  "args": [ ... runtime values ... ],
  "limits": {
    "instruction_limit": 100000,
    "stack_limit": 1024,
    "call_depth_limit": 128,
    "wall_time_ms": 200,
    "heap_limit_bytes": 16777216
  }
}
```

* `module` is the serialised `BytecodeModule` structure (see `rust/vm_runtime/src/bytecode.rs`).
* `entry` names the function to execute (defaults to `solve`).
* `args` is an array of VM values; container literals recursively mirror the
  `Value` enum (`Scalar`, `List`, `Map`, `Set`).
* `limits` is optional; omitted fields fall back to sandbox defaults.

A successful invocation returns:

```json
{
  "ok": true,
  "value": { ... Optional<Value> ... },
  "trace": {
    "trace": {
      "steps": [{"function": "solve", "pc": 0, "opcode": "LoadImmediate", "stack_depth": 0}],
      "coverage": {"solve": [0]},
      "metrics": {
        "instructions": 1,
        "unique_opcodes": ["LoadImmediate"],
        "loop_frames": 0,
        "loop_iterations": 0,
        "max_loop_depth": 0,
        "sandbox": {"instruction_count": 1, "max_stack_depth": 0, "max_call_depth": 1, "elapsed": 0},
        "heap": {"bytes_current": 0, "bytes_peak": 0, "allocations": 0}
      }
    },
    "sandbox": { ... duplicated for convenience ... },
    "heap": { ... duplicated ... }
  }
}
```

Errors are reported as `{"ok": false, "error": {"kind": "…", "message": "…", "detail": ...}}`.

## Trace fields

* **steps** – ordered instruction log with function name, program counter and
  stack depth at each step.
* **coverage** – executed instruction indices per function (deduplicated and
  sorted).
* **metrics** – aggregate counters:
  * `instructions`: total instructions retired.
  * `unique_opcodes`: set of opcodes executed at least once.
  * `loop_frames`, `loop_iterations`, `max_loop_depth`: derived from explicit
    `LoopEnter`/`LoopExit` markers.
  * `sandbox`: copy of sandbox metrics (instruction count, peak stack/call
    depth, elapsed wall time in milliseconds).
  * `heap`: conservative memory accounting (bytes currently used, peak bytes,
    allocation count).

## Sandbox controls

The sandbox enforces deterministic resource limits:

| Limit                | Default       | Notes                                           |
| -------------------- | ------------- | ----------------------------------------------- |
| `instruction_limit`  | 100,000       | Abort if exceeded.                              |
| `stack_limit`        | 1,024 values  | Applies per call frame.                         |
| `call_depth_limit`   | 128           | Maximum number of active frames.                |
| `wall_time_ms`       | 200 ms        | Soft wall-clock timeout (0 disables check).     |
| `heap_limit_bytes`   | 16 MiB        | Shared with the heap subsystem for allocations. |

The Python runner allows callers to override these thresholds via the `limits`
object; unspecified fields remain at their defaults.

## Builtins

The standard runtime exposes the following deterministic intrinsics via
`CallOperand::Intrinsic`:

| Name      | Description                                   |
| --------- | --------------------------------------------- |
| `identity`| Returns the single argument unchanged.        |
| `len`     | Length of list/map/set/string argument.       |
| `sum`     | Integer sum of a list of integers.            |
| `min`     | Minimum value in a non-empty int list.        |
| `max`     | Maximum value in a non-empty int list.        |
| `abs`     | Absolute value for int/float arguments.       |
| `not`     | Boolean negation.                             |

Additional intrinsics can be discovered programmatically via the
`praxis_vm_builtins` FFI helper.

## Memory model snapshot

* Scalars are immutable and copied by value.
* Containers (`List`, `Map`, `Set`) are lazily created: attempting to mutate an
  uninitialised register transparently instantiates the appropriate container.
* Heap metrics account for container overhead using compact heuristics (base and
  per-element costs).  The sandbox compares the current usage against the
  configured `heap_limit_bytes` after each mutation.

## Limitations (Omega revision)

* Control-flow lowering currently rejects branch and jump instructions; fixture
  programs should rely on straight-line code or pre-lowered bytecode.
* Indirect calls are unsupported.
* Unary operators in the IR are not yet lowered to bytecode.

Future revisions will remove these restrictions once the compiler pipeline is
complete.
