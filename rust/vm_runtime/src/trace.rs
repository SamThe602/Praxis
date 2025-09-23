//! Execution trace collection for the Praxis VM.
//!
//! The trace module gathers lightweight telemetry during interpretation so that
//! higher layers (verifier, orchestrator, UI) can reconstruct the execution
//! without rerunning the program.  The emphasis is on determinism and
//! predictability: we capture the order of executed opcodes, coverage within
//! each function, loop iteration counts, and aggregate metrics sourced from the
//! sandbox and heap subsystems.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bytecode::Opcode;
use crate::memory::MemoryMetrics;
use crate::sandbox::SandboxMetrics;

/// Trace emitted after execution completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub steps: Vec<TraceStep>,
    pub coverage: TraceCoverage,
    pub metrics: TraceMetrics,
}

/// Individual instruction execution event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub function: String,
    pub pc: usize,
    pub opcode: Opcode,
    pub stack_depth: usize,
}

/// Aggregated coverage report keyed by function name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCoverage {
    pub functions: BTreeMap<String, Vec<usize>>,
}

/// High-level metrics summarising the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetrics {
    pub instructions: u64,
    pub unique_opcodes: Vec<Opcode>,
    pub loop_frames: usize,
    pub loop_iterations: u64,
    pub max_loop_depth: usize,
    pub sandbox: SandboxMetrics,
    pub heap: MemoryMetrics,
}

impl TraceMetrics {
    fn new(
        instruction_count: u64,
        unique_opcodes: Vec<Opcode>,
        loop_frames: usize,
        loop_iterations: u64,
        max_loop_depth: usize,
        sandbox: SandboxMetrics,
        heap: MemoryMetrics,
    ) -> Self {
        Self {
            instructions: instruction_count,
            unique_opcodes,
            loop_frames,
            loop_iterations,
            max_loop_depth,
            sandbox,
            heap,
        }
    }
}

/// Internal helper recording trace state as the interpreter runs.
#[derive(Debug, Default)]
pub struct TraceCollector {
    steps: Vec<TraceStep>,
    coverage: BTreeMap<String, BTreeSet<usize>>,
    instruction_count: u64,
    unique_opcodes: BTreeSet<Opcode>,
    loop_frames: usize,
    loop_iterations: u64,
    max_loop_depth: usize,
    current_loop_depth: usize,
}

impl TraceCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_instruction(
        &mut self,
        function: &str,
        pc: usize,
        opcode: Opcode,
        stack_depth: usize,
    ) {
        self.instruction_count += 1;
        self.unique_opcodes.insert(opcode);
        self.steps.push(TraceStep {
            function: function.to_owned(),
            pc,
            opcode,
            stack_depth,
        });
        self.coverage
            .entry(function.to_owned())
            .or_default()
            .insert(pc);
    }

    pub fn loop_enter(&mut self) {
        self.loop_frames += 1;
        self.current_loop_depth += 1;
        if self.current_loop_depth > self.max_loop_depth {
            self.max_loop_depth = self.current_loop_depth;
        }
    }

    pub fn loop_exit(&mut self) {
        if self.current_loop_depth > 0 {
            self.current_loop_depth -= 1;
        }
    }

    pub fn loop_iteration(&mut self) {
        self.loop_iterations += 1;
    }

    pub fn finish(self, sandbox: SandboxMetrics, heap: &MemoryMetrics) -> ExecutionTrace {
        let coverage = TraceCoverage {
            functions: self
                .coverage
                .into_iter()
                .map(|(func, pcs)| (func, pcs.into_iter().collect()))
                .collect(),
        };
        let metrics = TraceMetrics::new(
            self.instruction_count,
            self.unique_opcodes.into_iter().collect(),
            self.loop_frames,
            self.loop_iterations,
            self.max_loop_depth,
            sandbox,
            heap.clone(),
        );
        ExecutionTrace {
            steps: self.steps,
            coverage,
            metrics,
        }
    }
}
