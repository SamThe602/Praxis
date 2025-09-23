//! Compilation entry points bridging IR and bytecode.
//!
//! The goal of this module is to provide a small, well-documented surface area
//! that the Python frontend can call into (via FFI bindings) once the lowering
//! pass is implemented.  For now the compiler merely mirrors function metadata
//! into the bytecode module, making it possible to unit-test downstream stages
//! without committing to a full optimiser.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::bytecode::{
    BytecodeFunction, BytecodeModule, CallOperand, ContractCheck, FunctionHandle, Immediate,
    Instruction, Opcode, Operands, Register, ScalarValue,
};
use crate::ir::{
    self, BinaryOp, CallSite, ContainerOp, FunctionContract, LoopInstruction, Operand,
};

/// Result alias for compilation stages.
pub type CompileResult<T> = Result<T, CompileError>;

/// Lightweight error type used across the lowering pipeline.
#[derive(Debug, Clone)]
pub struct CompileError {
    message: String,
}

impl CompileError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for CompileError {}

/// Source of IR modules.  The trait allows callers (including FFI shims) to
/// plug in custom loading strategies without modifying the compiler core.
pub trait ModuleSource {
    fn next_module(&mut self) -> CompileResult<Option<ir::IrModule>>;
}

/// Sink that receives fully lowered bytecode modules.
pub trait BytecodeSink {
    fn accept(&mut self, module: &BytecodeModule) -> CompileResult<()>;
}

/// High-level compiler orchestrating the lowering pipeline.
pub struct Compiler<S, T> {
    source: S,
    sink: T,
}

impl<S, T> Compiler<S, T>
where
    S: ModuleSource,
    T: BytecodeSink,
{
    pub fn new(source: S, sink: T) -> Self {
        Self { source, sink }
    }

    /// Drain the source and emit bytecode modules into the sink.
    pub fn run(&mut self) -> CompileResult<()> {
        while let Some(module) = self.source.next_module()? {
            let bytecode = lower_module(&module)?;
            self.sink.accept(&bytecode)?;
        }
        Ok(())
    }
}

/// Convenience helper for callers that already have an IR module in memory.
pub fn lower_module(module: &ir::IrModule) -> CompileResult<BytecodeModule> {
    let mut bytecode_module = BytecodeModule::new();
    bytecode_module.constants = module.constants.clone();
    let handles: HashMap<String, FunctionHandle> = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.name.clone(), FunctionHandle(index as u32)))
        .collect();
    for function in &module.functions {
        let lowered = lower_function(function, &handles)?;
        bytecode_module.push_function(lowered);
    }
    Ok(bytecode_module)
}

fn lower_function(
    function: &ir::IrFunction,
    handles: &HashMap<String, FunctionHandle>,
) -> CompileResult<BytecodeFunction> {
    let mut result = BytecodeFunction::new(&function.name);
    result.registers = function.register_span() as u16;
    result.stack_slots = function.stack_hint.unwrap_or(0);
    result.contracts = function.contracts.iter().map(lower_contract).collect();
    let mut instructions = Vec::new();
    for block in &function.blocks {
        for instruction in &block.instructions {
            instructions.extend(lower_instruction(instruction, handles)?);
        }
    }
    result.instructions = instructions;
    Ok(result)
}

fn lower_contract(contract: &FunctionContract) -> ContractCheck {
    ContractCheck {
        kind: contract.kind,
        binding: (&contract.annotation).into(),
    }
}

fn lower_instruction(
    instruction: &ir::IrInstruction,
    handles: &HashMap<String, FunctionHandle>,
) -> CompileResult<Vec<Instruction>> {
    use ir::IrInstruction as Ir;
    let mut emitted = Vec::new();
    match instruction {
        Ir::LoadImmediate { target, value } => emitted.push(Instruction::new(
            Opcode::LoadImmediate,
            Operands::RegImmediate(*target, Immediate::Scalar(value.clone())),
        )),
        Ir::LoadConst { target, constant } => emitted.push(Instruction::new(
            Opcode::LoadConst,
            Operands::RegImmediate(*target, Immediate::Constant(*constant)),
        )),
        Ir::Move { target, source } => match source {
            Operand::Register(register) => emitted.push(Instruction::new(
                Opcode::Move,
                Operands::RegPair(*target, *register),
            )),
            Operand::Immediate(value) => emitted.push(Instruction::new(
                Opcode::LoadImmediate,
                Operands::RegImmediate(*target, Immediate::Scalar(value.clone())),
            )),
            Operand::Constant(id) => emitted.push(Instruction::new(
                Opcode::LoadConst,
                Operands::RegImmediate(*target, Immediate::Constant(*id)),
            )),
            Operand::Stack(_) => {
                return Err(CompileError::new(
                    "stack operands are not supported by the lowering pipeline",
                ))
            }
        },
        Ir::Push { source } => emitted.push(Instruction::new(
            Opcode::Push,
            Operands::Reg(expect_register(source)?),
        )),
        Ir::Pop => emitted.push(Instruction::new(Opcode::Pop, Operands::None)),
        Ir::Binary {
            op,
            lhs,
            rhs,
            target,
        } => emitted.push(Instruction::new(
            map_binary_opcode(*op)?,
            Operands::RegTriple(*target, expect_register(lhs)?, expect_register(rhs)?),
        )),
        Ir::Unary { .. } => {
            return Err(CompileError::new(
                "unary operators are not supported by the current lowering",
            ))
        }
        Ir::Jump { .. } | Ir::Branch { .. } => {
            return Err(CompileError::new(
                "control-flow lowering is not implemented in the Omega revision",
            ))
        }
        Ir::Loop(loop_inst) => match loop_inst {
            LoopInstruction::Enter { .. } => {
                emitted.push(Instruction::new(Opcode::LoopEnter, Operands::None))
            }
            LoopInstruction::Continue { .. } => {
                emitted.push(Instruction::new(Opcode::LoopContinue, Operands::None))
            }
            LoopInstruction::Break { .. } => {
                emitted.push(Instruction::new(Opcode::LoopBreak, Operands::None))
            }
            LoopInstruction::Exit { .. } => {
                emitted.push(Instruction::new(Opcode::LoopExit, Operands::None))
            }
        },
        Ir::Call { site, args, result } => {
            for operand in args {
                emitted.push(Instruction::new(
                    Opcode::Push,
                    Operands::Reg(expect_register(operand)?),
                ));
            }
            let target = map_call_site(site, handles)?;
            emitted.push(Instruction::new(
                Opcode::Call,
                Operands::Call {
                    target,
                    arg_count: args.len() as u16,
                },
            ));
            match result {
                Some(register) => {
                    emitted.push(Instruction::new(Opcode::Pop, Operands::Reg(*register)))
                }
                None => emitted.push(Instruction::new(Opcode::Pop, Operands::None)),
            }
        }
        Ir::Container { op } => emitted.push(lower_container(op)?),
        Ir::Contract { kind, annotation } => emitted.push(Instruction::new(
            Opcode::AssertContract,
            Operands::Contract(ContractCheck {
                kind: *kind,
                binding: annotation.into(),
            }),
        )),
        Ir::Return { value } => match value {
            Some(operand) => emitted.push(Instruction::new(
                Opcode::Return,
                Operands::Reg(expect_register(operand)?),
            )),
            None => emitted.push(Instruction::new(Opcode::Return, Operands::None)),
        },
    }
    Ok(emitted)
}

fn expect_register(operand: &Operand) -> CompileResult<Register> {
    if let Operand::Register(register) = operand {
        Ok(*register)
    } else {
        Err(CompileError::new(
            "operand must be materialised in a register before lowering",
        ))
    }
}

fn map_call_site(
    site: &CallSite,
    handles: &HashMap<String, FunctionHandle>,
) -> CompileResult<CallOperand> {
    Ok(match site {
        CallSite::Direct(name) => {
            let handle = handles
                .get(name)
                .ok_or_else(|| CompileError::new(format!("unknown function '{name}'")))?;
            CallOperand::Direct(*handle)
        }
        CallSite::Intrinsic(name) => CallOperand::Intrinsic(name.clone()),
        CallSite::Indirect(_) => {
            return Err(CompileError::new(
                "indirect calls are not supported by the current lowering",
            ))
        }
    })
}

fn map_binary_opcode(op: BinaryOp) -> CompileResult<Opcode> {
    Ok(match op {
        BinaryOp::Add => Opcode::Add,
        BinaryOp::Sub => Opcode::Sub,
        BinaryOp::Mul => Opcode::Mul,
        BinaryOp::Div => Opcode::Div,
        BinaryOp::Mod => Opcode::Mod,
        BinaryOp::Eq => Opcode::Eq,
        BinaryOp::Ne => Opcode::Ne,
        BinaryOp::Lt => Opcode::Lt,
        BinaryOp::Le => Opcode::Le,
        BinaryOp::Gt => Opcode::Gt,
        BinaryOp::Ge => Opcode::Ge,
        BinaryOp::And => Opcode::And,
        BinaryOp::Or => Opcode::Or,
    })
}

fn lower_container(op: &ContainerOp) -> CompileResult<Instruction> {
    Ok(match op {
        ContainerOp::ListNew {
            target,
            initial_arity,
        } => Instruction::new(
            Opcode::ListNew,
            Operands::RegImmediate(
                *target,
                Immediate::Scalar(ScalarValue::Int(*initial_arity as i64)),
            ),
        ),
        ContainerOp::ListPush { list, value } => Instruction::new(
            Opcode::ListPush,
            Operands::RegPair(*list, expect_register(value)?),
        ),
        ContainerOp::ListPop { list, target } => {
            Instruction::new(Opcode::ListPop, Operands::RegPair(*list, *target))
        }
        ContainerOp::MapInsert { map, key, value } => Instruction::new(
            Opcode::MapInsert,
            Operands::RegTriple(*map, expect_register(key)?, expect_register(value)?),
        ),
        ContainerOp::MapGet { map, key, target } => Instruction::new(
            Opcode::MapGet,
            Operands::RegTriple(*map, expect_register(key)?, *target),
        ),
        ContainerOp::SetInsert { set, value } => Instruction::new(
            Opcode::SetInsert,
            Operands::RegPair(*set, expect_register(value)?),
        ),
        ContainerOp::Length { target, container } => Instruction::new(
            Opcode::Length,
            Operands::RegPair(*target, expect_register(container)?),
        ),
    })
}
