//! Bytecode model for the Praxis virtual machine.
//!
//! The VM executes a compact instruction stream that interleaves register-based
//! operations with an evaluation stack for expression temporaries.  The types
//! below describe the bytecode module format, instruction set, and supporting
//! metadata used by the runtime and disassembler.  They intentionally mirror
//! the mid-level IR so lowering remains largely mechanical.

use serde::{Deserialize, Serialize};

/// General purpose register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Register(pub u16);

/// Stack slot index used for transient values or spilled registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackIndex(pub u16);

/// Handle into the constant pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstantId(pub u32);

/// Handle referencing a function within the bytecode module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionHandle(pub u32);

/// Scalar literal that can be embedded directly in the instruction stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Unit,
}

/// Immediate operand used by instructions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Immediate {
    Scalar(ScalarValue),
    Constant(ConstantId),
}

/// Classification of contract assertions emitted by the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContractKind {
    Pre,
    Post,
    Requires,
    Ensures,
}

/// Resource binding associated with a contract assertion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceBinding {
    pub resource: String,
    pub detail: Option<String>,
}

/// Module-level container for bytecode functions and constants.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BytecodeModule {
    pub constants: Vec<ScalarValue>,
    pub functions: Vec<BytecodeFunction>,
}

impl BytecodeModule {
    /// Construct an empty module.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function definition to the module.
    pub fn push_function(&mut self, function: BytecodeFunction) {
        self.functions.push(function);
    }
}

/// Bytecode function along with execution metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BytecodeFunction {
    pub name: String,
    pub registers: u16,
    pub stack_slots: u16,
    pub instructions: Vec<Instruction>,
    pub contracts: Vec<ContractCheck>,
}

impl BytecodeFunction {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            registers: 0,
            stack_slots: 0,
            instructions: Vec::new(),
            contracts: Vec::new(),
        }
    }

    /// Reserve storage for instructions to avoid reallocations during lowering.
    pub fn with_capacity(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            registers: 0,
            stack_slots: 0,
            instructions: Vec::with_capacity(capacity),
            contracts: Vec::new(),
        }
    }

    /// Append an instruction to the function body.
    pub fn push(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }
}

/// Contract check inserted into the bytecode stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContractCheck {
    pub kind: ContractKind,
    pub binding: ResourceBinding,
}

/// Primary instruction format for the VM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: Opcode,
    pub operands: Operands,
}

impl Instruction {
    pub fn new(opcode: Opcode, operands: Operands) -> Self {
        Self { opcode, operands }
    }
}

/// Enumeration of all opcodes recognised by the VM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Opcode {
    /// Load a scalar literal into a register.
    LoadImmediate,
    /// Load a constant pool entry into a register.
    LoadConst,
    /// Copy between registers.
    Move,
    /// Push a register onto the evaluation stack.
    Push,
    /// Pop the top of the evaluation stack.
    Pop,
    /// Arithmetic and logical operations.
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    /// Unconditional jump with relative offset.
    Jump,
    /// Conditional jump if the top of stack evaluates to false.
    JumpIfFalse,
    /// Call another function (direct or indirect).
    Call,
    /// Return to the caller.
    Return,
    /// Mark the start of a loop frame.
    LoopEnter,
    /// Mark the end of a loop frame.
    LoopExit,
    /// Continue to the loop header.
    LoopContinue,
    /// Break out of the loop frame.
    LoopBreak,
    /// Allocate a new list on the heap.
    ListNew,
    /// Append to a list.
    ListPush,
    /// Pop from a list.
    ListPop,
    /// Insert into a map.
    MapInsert,
    /// Lookup from a map.
    MapGet,
    /// Insert into a set.
    SetInsert,
    /// Query the length of a container.
    Length,
    /// Runtime contract assertion.
    AssertContract,
}

/// Structured operand payload accompanying an opcode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum Operands {
    None,
    Reg(Register),
    RegPair(Register, Register),
    RegTriple(Register, Register, Register),
    RegImmediate(Register, Immediate),
    Stack(StackIndex),
    Jump { offset: i32 },
    Call { target: CallOperand, arg_count: u16 },
    Contract(ContractCheck),
}

/// Operand describing the call target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum CallOperand {
    Direct(FunctionHandle),
    Indirect(Register),
    Intrinsic(String),
}
