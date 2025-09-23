# Explanation Examples

The explanation engine produces concise Markdown narratives from the program,
trace, and verifier evidence. The snippets below illustrate the default style.

## Successful Verification

```
### Summary
Program `solve` for `array_sort` returned {'result': [1, 2, 3]}; verifier status: ok.

### Trace Highlights
- Step 1: load_input – pulled `arr` into registers.
- Step 2: sort_values – produced a sorted view while tracking swaps.

### Verification
- contract_check: passed (post-condition honoured).

### Telemetry
- praxis.vm.latency_ms: 4.2.
- praxis.synth.search_nodes: 18.

```

## Verification Failure

```
### Summary
Program `solve` for `matrix_mult` returned None; verifier status: failed.

### Trace Highlights
- Step 1: validate_dimensions – matrix B reported incompatible dimensions.

### Verification
- shape_check: failed (expected 2x2, received 2x3).
- Failure: result registers not populated.

### Caveats
- Verification reported issues; review recommended.

```
