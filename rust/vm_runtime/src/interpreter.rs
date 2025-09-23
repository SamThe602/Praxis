//! Bytecode interpreter for the Praxis virtual machine.
//!
//! The interpreter executes deterministic bytecode produced by the compiler.
//! It is intentionally conservative: all memory allocations go through the
//! tracked heap, the sandbox enforces instruction/stack/time limits, and every
//! executed opcode is recorded by the trace collector for downstream analysis.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;

use thiserror::Error;

use crate::builtins::{BuiltinError, Builtins};
use crate::bytecode::{
    BytecodeFunction, BytecodeModule, CallOperand, ConstantId, FunctionHandle, Immediate,
    Instruction, Opcode, Operands, Register, ScalarValue,
};
use crate::memory::{Heap, MemoryError, Value, ValueKey};
use crate::sandbox::{Sandbox, SandboxError, SandboxLimits};
use crate::trace::{ExecutionTrace, TraceCollector};

/// Configuration controlling interpreter resource budgets.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    pub sandbox_limits: SandboxLimits,
    pub heap_limit_bytes: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        let sandbox_limits = SandboxLimits::default();
        Self {
            heap_limit_bytes: sandbox_limits.heap_limit_bytes,
            sandbox_limits,
        }
    }
}

/// Result of executing a function.
#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    pub return_value: Option<Value>,
    pub trace: ExecutionTrace,
}

/// Public interpreter facade reused by the FFI layer.
#[derive(Debug, Clone)]
pub struct Interpreter {
    module: BytecodeModule,
    builtins: Builtins,
    config: ExecutionConfig,
}

impl Interpreter {
    pub fn new(module: BytecodeModule) -> Self {
        Self::with_config(module, ExecutionConfig::default())
    }

    pub fn with_config(module: BytecodeModule, config: ExecutionConfig) -> Self {
        Self {
            module,
            builtins: Builtins::standard(),
            config,
        }
    }

    pub fn with_builtins(
        module: BytecodeModule,
        builtins: Builtins,
        config: ExecutionConfig,
    ) -> Self {
        Self {
            module,
            builtins,
            config,
        }
    }

    pub fn execute(&self, entry: &str, args: Vec<Value>) -> Result<ExecutionOutcome, VmError> {
        VmState::new(&self.module, &self.builtins, &self.config).execute(entry, args)
    }
}

struct VmState<'a> {
    module: &'a BytecodeModule,
    builtins: &'a Builtins,
    sandbox: Sandbox,
    heap: Heap,
    trace: TraceCollector,
}

impl<'a> VmState<'a> {
    fn new(module: &'a BytecodeModule, builtins: &'a Builtins, config: &ExecutionConfig) -> Self {
        Self {
            module,
            builtins,
            sandbox: Sandbox::new(config.sandbox_limits),
            heap: Heap::new(config.heap_limit_bytes),
            trace: TraceCollector::new(),
        }
    }

    fn execute(mut self, entry: &str, args: Vec<Value>) -> Result<ExecutionOutcome, VmError> {
        let entry_index = self
            .module
            .functions
            .iter()
            .position(|function| function.name == entry)
            .ok_or_else(|| VmError::FunctionNotFound(entry.to_owned()))?;

        let mut frames = Vec::new();
        let entry_function = &self.module.functions[entry_index];
        frames.push(Frame::new(entry_index, entry_function, args)?);
        self.sandbox.record_call_depth(frames.len())?;

        let mut final_value: Option<Value> = None;

        loop {
            if frames.is_empty() {
                break;
            }

            self.sandbox.observe_instruction()?;

            // Fetch instruction metadata without holding a borrow across the
            // subsequent match block.
            let (pc, instruction, stack_depth, function_name) = {
                let frame = frames.last_mut().expect("frame stack checked above");
                let function = &self.module.functions[frame.function_index];
                if frame.pc >= function.instructions.len() {
                    // Implicit return (no explicit `Return` encountered).
                    let value = Value::unit();
                    frames.pop();
                    self.sandbox.record_call_depth(frames.len())?;
                    if let Some(parent) = frames.last_mut() {
                        parent.stack.push(value.clone());
                        self.sandbox.record_stack_depth(parent.stack.len())?;
                        continue;
                    } else {
                        final_value = Some(value);
                        break;
                    }
                }
                let pc = frame.pc;
                let instruction = function.instructions[pc].clone();
                frame.pc += 1;
                let stack_depth = frame.stack.len();
                (pc, instruction, stack_depth, function.name.clone())
            };

            self.trace
                .record_instruction(&function_name, pc, instruction.opcode, stack_depth);

            let frame = frames
                .last_mut()
                .expect("frame present for instruction execution");

            match instruction.opcode {
                Opcode::LoadImmediate => {
                    if let Operands::RegImmediate(target, immediate) = instruction.operands {
                        let value = self.immediate_to_value(&immediate)?;
                        frame.write_register(target, value);
                    } else {
                        return Err(VmError::InvalidOperands("LoadImmediate"));
                    }
                }
                Opcode::LoadConst => {
                    if let Operands::RegImmediate(target, immediate) = instruction.operands {
                        let constant = match immediate {
                            Immediate::Constant(id) => self.load_constant(id)?,
                            _ => return Err(VmError::InvalidOperands("LoadConst")),
                        };
                        frame.write_register(target, constant);
                    } else {
                        return Err(VmError::InvalidOperands("LoadConst"));
                    }
                }
                Opcode::Move => {
                    if let Operands::RegPair(dst, src) = instruction.operands {
                        let value = frame.read_register(src)?;
                        frame.write_register(dst, value);
                    } else {
                        return Err(VmError::InvalidOperands("Move"));
                    }
                }
                Opcode::Push => {
                    if let Operands::Reg(reg) = instruction.operands {
                        let value = frame.read_register(reg)?;
                        frame.stack.push(value);
                        self.sandbox.record_stack_depth(frame.stack.len())?;
                    } else {
                        return Err(VmError::InvalidOperands("Push"));
                    }
                }
                Opcode::Pop => match instruction.operands {
                    Operands::None => {
                        frame.stack.pop().ok_or(VmError::StackUnderflow("Pop"))?;
                        self.sandbox.record_stack_depth(frame.stack.len())?;
                    }
                    Operands::Reg(target) => {
                        let value = frame.stack.pop().ok_or(VmError::StackUnderflow("Pop"))?;
                        frame.write_register(target, value);
                        self.sandbox.record_stack_depth(frame.stack.len())?;
                    }
                    _ => return Err(VmError::InvalidOperands("Pop")),
                },
                Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Mod
                | Opcode::Eq
                | Opcode::Ne
                | Opcode::Lt
                | Opcode::Le
                | Opcode::Gt
                | Opcode::Ge
                | Opcode::And
                | Opcode::Or => self.execute_binary(frame, instruction)?,
                Opcode::Jump => {
                    if let Operands::Jump { offset } = instruction.operands {
                        apply_jump(&mut frames, offset)?;
                    } else {
                        return Err(VmError::InvalidOperands("Jump"));
                    }
                }
                Opcode::JumpIfFalse => {
                    if let Operands::Jump { offset } = instruction.operands {
                        let condition = frame
                            .stack
                            .pop()
                            .ok_or(VmError::StackUnderflow("JumpIfFalse"))?;
                        self.sandbox.record_stack_depth(frame.stack.len())?;
                        let is_true = value_to_bool(&condition)?;
                        if !is_true {
                            apply_jump(&mut frames, offset)?;
                        }
                    } else {
                        return Err(VmError::InvalidOperands("JumpIfFalse"));
                    }
                }
                Opcode::Call => {
                    self.execute_call(&mut frames, instruction)?;
                }
                Opcode::Return => {
                    let maybe_value = match instruction.operands {
                        Operands::Reg(reg) => Some(frame.read_register(reg)?),
                        Operands::None => None,
                        _ => return Err(VmError::InvalidOperands("Return")),
                    };
                    let return_value = maybe_value.unwrap_or_else(Value::unit);
                    frames.pop();
                    self.sandbox.record_call_depth(frames.len())?;
                    if let Some(parent) = frames.last_mut() {
                        parent.stack.push(return_value.clone());
                        self.sandbox.record_stack_depth(parent.stack.len())?;
                    } else {
                        final_value = Some(return_value);
                        break;
                    }
                }
                Opcode::LoopEnter => {
                    self.trace.loop_enter();
                }
                Opcode::LoopExit => {
                    self.trace.loop_exit();
                }
                Opcode::LoopContinue => {
                    self.trace.loop_iteration();
                }
                Opcode::LoopBreak => {
                    self.trace.loop_iteration();
                    self.trace.loop_exit();
                }
                Opcode::ListNew => {
                    if let Operands::RegImmediate(
                        target,
                        Immediate::Scalar(ScalarValue::Int(initial)),
                    ) = instruction.operands
                    {
                        self.heap.register_container_allocation()?;
                        let capacity = if initial.is_negative() {
                            0
                        } else {
                            initial as usize
                        };
                        frame.write_register(target, Value::List(Vec::with_capacity(capacity)));
                        self.sandbox
                            .enforce_heap_usage(self.heap.metrics().bytes_current)?;
                    } else {
                        return Err(VmError::InvalidOperands("ListNew"));
                    }
                }
                Opcode::ListPush => {
                    if let Operands::RegPair(list_reg, value_reg) = instruction.operands {
                        let value = frame.read_register(value_reg)?;
                        let slot = frame.register_slot_mut(list_reg)?;
                        if !matches!(slot, Value::List(_)) {
                            self.heap.register_container_allocation()?;
                            *slot = Value::List(Vec::new());
                        }
                        if let Value::List(items) = slot {
                            self.heap.list_grew()?;
                            items.push(value);
                        }
                        self.sandbox
                            .enforce_heap_usage(self.heap.metrics().bytes_current)?;
                    } else {
                        return Err(VmError::InvalidOperands("ListPush"));
                    }
                }
                Opcode::ListPop => {
                    if let Operands::RegPair(list_reg, target_reg) = instruction.operands {
                        let slot = frame.register_slot_mut(list_reg)?;
                        if let Value::List(items) = slot {
                            let value = items
                                .pop()
                                .ok_or(VmError::InvalidState("ListPop on empty list"))?;
                            self.heap.list_shrank();
                            frame.write_register(target_reg, value);
                        } else {
                            return Err(VmError::TypeMismatch {
                                expected: "list",
                                found: slot.kind(),
                            });
                        }
                    } else {
                        return Err(VmError::InvalidOperands("ListPop"));
                    }
                }
                Opcode::MapInsert => {
                    if let Operands::RegTriple(map_reg, key_reg, value_reg) = instruction.operands {
                        let key_value = frame.read_register(key_reg)?;
                        let value = frame.read_register(value_reg)?;
                        let slot = frame.register_slot_mut(map_reg)?;
                        if !matches!(slot, Value::Map(_)) {
                            self.heap.register_container_allocation()?;
                            *slot = Value::Map(BTreeMap::new());
                        }
                        if let Value::Map(entries) = slot {
                            let key = ValueKey::try_from(&key_value)?;
                            if entries.insert(key, value).is_none() {
                                self.heap.map_grew()?;
                            }
                        }
                        self.sandbox
                            .enforce_heap_usage(self.heap.metrics().bytes_current)?;
                    } else {
                        return Err(VmError::InvalidOperands("MapInsert"));
                    }
                }
                Opcode::MapGet => {
                    if let Operands::RegTriple(map_reg, key_reg, target_reg) = instruction.operands
                    {
                        let key_value = frame.read_register(key_reg)?;
                        let fetched = {
                            let slot = frame.register_slot(map_reg)?;
                            match slot {
                                Value::Map(entries) => {
                                    let key = ValueKey::try_from(&key_value)?;
                                    entries
                                        .get(&key)
                                        .cloned()
                                        .ok_or(VmError::InvalidState("Map key missing"))?
                                }
                                other => {
                                    return Err(VmError::TypeMismatch {
                                        expected: "map",
                                        found: other.kind(),
                                    })
                                }
                            }
                        };
                        frame.write_register(target_reg, fetched);
                    } else {
                        return Err(VmError::InvalidOperands("MapGet"));
                    }
                }
                Opcode::SetInsert => {
                    if let Operands::RegPair(set_reg, value_reg) = instruction.operands {
                        let value = frame.read_register(value_reg)?;
                        let slot = frame.register_slot_mut(set_reg)?;
                        if !matches!(slot, Value::Set(_)) {
                            self.heap.register_container_allocation()?;
                            *slot = Value::Set(BTreeSet::new());
                        }
                        if let Value::Set(entries) = slot {
                            let key = ValueKey::try_from(&value)?;
                            if entries.insert(key) {
                                self.heap.set_grew()?;
                            }
                        }
                        self.sandbox
                            .enforce_heap_usage(self.heap.metrics().bytes_current)?;
                    } else {
                        return Err(VmError::InvalidOperands("SetInsert"));
                    }
                }
                Opcode::Length => {
                    if let Operands::RegPair(target_reg, source_reg) = instruction.operands {
                        let value = frame.read_register(source_reg)?;
                        let len = value.len().ok_or(VmError::TypeMismatch {
                            expected: "len-compatible",
                            found: value.kind(),
                        })? as i64;
                        frame.write_register(target_reg, ScalarValue::Int(len).into());
                    } else {
                        return Err(VmError::InvalidOperands("Length"));
                    }
                }
                Opcode::AssertContract => {
                    if let Operands::Contract(binding) = instruction.operands {
                        let condition = frame
                            .stack
                            .pop()
                            .ok_or(VmError::StackUnderflow("AssertContract"))?;
                        self.sandbox.record_stack_depth(frame.stack.len())?;
                        let ok = value_to_bool(&condition)?;
                        if !ok {
                            return Err(VmError::ContractViolated(binding));
                        }
                    } else {
                        return Err(VmError::InvalidOperands("AssertContract"));
                    }
                }
            }
        }

        let trace = self
            .trace
            .finish(self.sandbox.metrics(), self.heap.metrics());

        Ok(ExecutionOutcome {
            return_value: final_value,
            trace,
        })
    }

    fn load_constant(&self, id: ConstantId) -> Result<Value, VmError> {
        let index = id.0 as usize;
        let value = self
            .module
            .constants
            .get(index)
            .ok_or(VmError::ConstantOutOfBounds(index))?;
        Ok(value.clone().into())
    }

    fn immediate_to_value(&self, immediate: &Immediate) -> Result<Value, VmError> {
        match immediate {
            Immediate::Scalar(value) => Ok(value.clone().into()),
            Immediate::Constant(id) => self.load_constant(*id),
        }
    }

    fn execute_call(
        &mut self,
        frames: &mut Vec<Frame>,
        instruction: Instruction,
    ) -> Result<(), VmError> {
        let (target, arg_count) = match instruction.operands {
            Operands::Call { target, arg_count } => (target, arg_count),
            _ => return Err(VmError::InvalidOperands("Call")),
        };

        let frame = frames.last_mut().expect("frame exists during call");

        if frame.stack.len() < arg_count as usize {
            return Err(VmError::StackUnderflow("Call arguments"));
        }

        let mut args = Vec::with_capacity(arg_count as usize);
        for _ in 0..arg_count {
            args.push(frame.stack.pop().expect("checked length"));
        }
        args.reverse();
        self.sandbox.record_stack_depth(frame.stack.len())?;

        match target {
            CallOperand::Intrinsic(name) => {
                let result = self.builtins.call(&name, &args)?;
                frame.stack.push(result);
                self.sandbox.record_stack_depth(frame.stack.len())?;
            }
            CallOperand::Direct(handle) => {
                let index = self.resolve_handle(handle)?;
                let function = &self.module.functions[index];
                let mut new_frame = Frame::new(index, function, args)?;
                // Pre-allocate stack capacity hint if provided.
                if function.stack_slots > 0 {
                    new_frame.stack.reserve(function.stack_slots as usize);
                }
                frames.push(new_frame);
                self.sandbox.record_call_depth(frames.len())?;
            }
            CallOperand::Indirect(_) => {
                return Err(VmError::Unsupported("Indirect calls"));
            }
        }

        Ok(())
    }

    fn execute_binary(
        &mut self,
        frame: &mut Frame,
        instruction: Instruction,
    ) -> Result<(), VmError> {
        let (dst, lhs_reg, rhs_reg) = match instruction.operands {
            Operands::RegTriple(dst, lhs, rhs) => (dst, lhs, rhs),
            _ => return Err(VmError::InvalidOperands("binary")),
        };
        let lhs = frame.read_register(lhs_reg)?;
        let rhs = frame.read_register(rhs_reg)?;
        let result = match instruction.opcode {
            Opcode::Add => arithmetic_add(&lhs, &rhs)?,
            Opcode::Sub => arithmetic_sub(&lhs, &rhs)?,
            Opcode::Mul => arithmetic_mul(&lhs, &rhs)?,
            Opcode::Div => arithmetic_div(&lhs, &rhs)?,
            Opcode::Mod => arithmetic_mod(&lhs, &rhs)?,
            Opcode::Eq => ScalarValue::Bool(lhs == rhs).into(),
            Opcode::Ne => ScalarValue::Bool(lhs != rhs).into(),
            Opcode::Lt => compare_ordering(&lhs, &rhs, |ord| ord.is_lt())?,
            Opcode::Le => compare_ordering(&lhs, &rhs, |ord| ord.is_le())?,
            Opcode::Gt => compare_ordering(&lhs, &rhs, |ord| ord.is_gt())?,
            Opcode::Ge => compare_ordering(&lhs, &rhs, |ord| ord.is_ge())?,
            Opcode::And => logical_and(&lhs, &rhs)?,
            Opcode::Or => logical_or(&lhs, &rhs)?,
            _ => return Err(VmError::InvalidOperands("binary opcode")),
        };
        frame.write_register(dst, result);
        Ok(())
    }

    fn resolve_handle(&self, handle: FunctionHandle) -> Result<usize, VmError> {
        let index = handle.0 as usize;
        if index < self.module.functions.len() {
            Ok(index)
        } else {
            Err(VmError::FunctionHandleOutOfBounds(index))
        }
    }
}

#[derive(Debug)]
struct Frame {
    function_index: usize,
    registers: Vec<Value>,
    stack: Vec<Value>,
    pc: usize,
}

impl Frame {
    fn new(
        function_index: usize,
        function: &BytecodeFunction,
        args: Vec<Value>,
    ) -> Result<Self, VmError> {
        let register_count = function.registers as usize;
        if args.len() > register_count {
            return Err(VmError::ArityMismatch {
                function: function.name.clone(),
                expected: register_count,
                actual: args.len(),
            });
        }
        let mut registers = vec![Value::unit(); register_count];
        for (idx, value) in args.into_iter().enumerate() {
            registers[idx] = value;
        }
        Ok(Self {
            function_index,
            registers,
            stack: Vec::new(),
            pc: 0,
        })
    }

    fn read_register(&self, register: Register) -> Result<Value, VmError> {
        let index = register.0 as usize;
        self.registers
            .get(index)
            .cloned()
            .ok_or(VmError::InvalidRegister(index))
    }

    fn write_register(&mut self, register: Register, value: Value) {
        let index = register.0 as usize;
        if index >= self.registers.len() {
            self.registers.resize(index + 1, Value::unit());
        }
        self.registers[index] = value;
    }

    fn register_slot(&self, register: Register) -> Result<&Value, VmError> {
        let index = register.0 as usize;
        self.registers
            .get(index)
            .ok_or(VmError::InvalidRegister(index))
    }

    fn register_slot_mut(&mut self, register: Register) -> Result<&mut Value, VmError> {
        let index = register.0 as usize;
        if index >= self.registers.len() {
            self.registers.resize(index + 1, Value::unit());
        }
        self.registers
            .get_mut(index)
            .ok_or(VmError::InvalidRegister(index))
    }
}

fn apply_jump(frames: &mut [Frame], offset: i32) -> Result<(), VmError> {
    let frame = frames.last_mut().expect("frame present for jump");
    let pc = frame.pc as i64 + offset as i64;
    if pc < 0 {
        return Err(VmError::InvalidState("jump before function start"));
    }
    frame.pc = pc as usize;
    Ok(())
}

fn value_to_bool(value: &Value) -> Result<bool, VmError> {
    if let Some(boolean) = value.as_bool() {
        Ok(boolean)
    } else {
        Err(VmError::TypeMismatch {
            expected: "bool",
            found: value.kind(),
        })
    }
}

fn arithmetic_add(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    numeric_binary(lhs, rhs, |a, b| a + b, |a, b| a + b)
}

fn arithmetic_sub(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    numeric_binary(lhs, rhs, |a, b| a - b, |a, b| a - b)
}

fn arithmetic_mul(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    numeric_binary(lhs, rhs, |a, b| a * b, |a, b| a * b)
}

fn arithmetic_div(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    match (lhs.as_int(), rhs.as_int()) {
        (Some(a), Some(b)) => {
            if b == 0 {
                return Err(VmError::InvalidState("divide by zero"));
            }
            Ok(ScalarValue::Int(a / b).into())
        }
        _ => numeric_binary(lhs, rhs, |a, b| a / b, |a, b| a / b),
    }
}

fn arithmetic_mod(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    match (lhs.as_int(), rhs.as_int()) {
        (Some(a), Some(b)) => {
            if b == 0 {
                return Err(VmError::InvalidState("modulo by zero"));
            }
            Ok(ScalarValue::Int(a % b).into())
        }
        _ => Err(VmError::TypeMismatch {
            expected: "int",
            found: "non-int",
        }),
    }
}

fn numeric_binary(
    lhs: &Value,
    rhs: &Value,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, VmError> {
    match (lhs, rhs) {
        (Value::Scalar(ScalarValue::Int(a)), Value::Scalar(ScalarValue::Int(b))) => {
            Ok(ScalarValue::Int(int_op(*a, *b)).into())
        }
        (Value::Scalar(ScalarValue::Float(a)), Value::Scalar(ScalarValue::Float(b))) => {
            Ok(ScalarValue::Float(float_op(*a, *b)).into())
        }
        (Value::Scalar(ScalarValue::Int(a)), Value::Scalar(ScalarValue::Float(b))) => {
            Ok(ScalarValue::Float(float_op(*a as f64, *b)).into())
        }
        (Value::Scalar(ScalarValue::Float(a)), Value::Scalar(ScalarValue::Int(b))) => {
            Ok(ScalarValue::Float(float_op(*a, *b as f64)).into())
        }
        _ => Err(VmError::TypeMismatch {
            expected: "numeric",
            found: lhs.kind(),
        }),
    }
}

fn compare_ordering(
    lhs: &Value,
    rhs: &Value,
    predicate: impl Fn(Ordering) -> bool,
) -> Result<Value, VmError> {
    match (lhs, rhs) {
        (Value::Scalar(ScalarValue::Int(a)), Value::Scalar(ScalarValue::Int(b))) => {
            Ok(ScalarValue::Bool(predicate(a.cmp(b))).into())
        }
        (Value::Scalar(ScalarValue::Float(a)), Value::Scalar(ScalarValue::Float(b))) => {
            Ok(ScalarValue::Bool(predicate(
                a.partial_cmp(b)
                    .ok_or(VmError::InvalidState("NaN comparison"))?,
            ))
            .into())
        }
        (Value::Scalar(ScalarValue::String(a)), Value::Scalar(ScalarValue::String(b))) => {
            Ok(ScalarValue::Bool(predicate(a.cmp(b))).into())
        }
        _ => Err(VmError::TypeMismatch {
            expected: "comparable",
            found: lhs.kind(),
        }),
    }
}

fn logical_and(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    let a = value_to_bool(lhs)?;
    let b = value_to_bool(rhs)?;
    Ok(ScalarValue::Bool(a && b).into())
}

fn logical_or(lhs: &Value, rhs: &Value) -> Result<Value, VmError> {
    let a = value_to_bool(lhs)?;
    let b = value_to_bool(rhs)?;
    Ok(ScalarValue::Bool(a || b).into())
}
/// Errors surfaced by the interpreter.
#[derive(Debug, Error)]
pub enum VmError {
    #[error("function '{0}' not found in module")]
    FunctionNotFound(String),
    #[error("function index {0} out of bounds for handle lookup")]
    FunctionHandleOutOfBounds(usize),
    #[error("constant index {0} out of bounds")]
    ConstantOutOfBounds(usize),
    #[error("invalid register index {0}")]
    InvalidRegister(usize),
    #[error("instruction attempted to pop from empty stack during {0}")]
    StackUnderflow(&'static str),
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        expected: &'static str,
        found: &'static str,
    },
    #[error("invalid operands for opcode {0}")]
    InvalidOperands(&'static str),
    #[error("invalid state: {0}")]
    InvalidState(&'static str),
    #[error("unsupported feature: {0}")]
    Unsupported(&'static str),
    #[error("arity mismatch for function {function}: expected {expected}, received {actual}")]
    ArityMismatch {
        function: String,
        expected: usize,
        actual: usize,
    },
    #[error("built-in error: {0}")]
    Builtin(#[from] BuiltinError),
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    #[error("sandbox violation: {0}")]
    Sandbox(#[from] SandboxError),
    #[error("contract violated: {0:?}")]
    ContractViolated(crate::bytecode::ContractCheck),
}
