//! Mid-level intermediate representation for the Praxis VM compiler pipeline.
//!
//! The IR models control flow and effects at a granularity that is convenient
//! for traditional SSA-style optimisations while still reflecting the eventual
//! stack/register hybrid machine model.  Lowering from the Python DSL will
//! produce this IR before handing off to the bytecode encoder.  The structures
//! defined here intentionally trade exhaustiveness for clarity: every enum and
//! struct carries enough information for future passes without committing to a
//! particular optimisation strategy ahead of time.

use crate::bytecode::{
    ConstantId, ContractKind, Register, ResourceBinding, ScalarValue, StackIndex,
};

/// Identifier used to reference functions within a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

/// Identifier used to reference basic blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// Mid-level module consisting of functions and a shared constant pool.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IrModule {
    /// Collection of functions in source order.
    pub functions: Vec<IrFunction>,
    /// Literal constants shared across the module.
    pub constants: Vec<ScalarValue>,
}

impl IrModule {
    /// Construct an empty module.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a function definition.
    pub fn add_function(&mut self, function: IrFunction) {
        self.functions.push(function);
    }
}

/// Function level metadata and body.
#[derive(Debug, Clone, PartialEq)]
pub struct IrFunction {
    /// Unique function name (closely mirrors DSL symbol name).
    pub name: String,
    /// Ordered list of parameters.
    pub params: Vec<IrParameter>,
    /// Local registers allocated after parameters.
    pub locals: Vec<IrLocal>,
    /// Entry block identifier; every function must have at least one block.
    pub entry: BlockId,
    /// All basic blocks belonging to the function.
    pub blocks: Vec<IrBlock>,
    /// Resource and contract metadata harvested from decorators.
    pub contracts: Vec<FunctionContract>,
    /// Optional stack depth hint emitted by the high-level frontend.
    pub stack_hint: Option<u16>,
}

impl IrFunction {
    /// Derive the total register footprint (parameters + locals).
    pub fn register_span(&self) -> usize {
        self.params.len() + self.locals.len()
    }
}

/// Function parameter mapped to an explicit register.
#[derive(Debug, Clone, PartialEq)]
pub struct IrParameter {
    pub name: String,
    pub register: Register,
}

/// Local binding mapped to an explicit register.
#[derive(Debug, Clone, PartialEq)]
pub struct IrLocal {
    pub name: String,
    pub register: Register,
}

/// A resource annotation attached to a contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceAnnotation {
    /// Canonical resource name (e.g. "cpu", "memory:gpu").
    pub resource: String,
    /// Structured detail field used for quantities or scopes.
    pub detail: Option<String>,
}

impl From<&ResourceAnnotation> for ResourceBinding {
    fn from(annotation: &ResourceAnnotation) -> Self {
        Self {
            resource: annotation.resource.clone(),
            detail: annotation.detail.clone(),
        }
    }
}

/// Summary of a contract attached to a function definition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionContract {
    pub kind: ContractKind,
    pub annotation: ResourceAnnotation,
}

/// A basic block made up of sequential instructions.
#[derive(Debug, Clone, PartialEq)]
pub struct IrBlock {
    pub id: BlockId,
    pub instructions: Vec<IrInstruction>,
}

impl IrBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            instructions: Vec::new(),
        }
    }

    pub fn push(&mut self, instruction: IrInstruction) {
        self.instructions.push(instruction);
    }
}

/// Reference to a value that can be consumed by an instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// General purpose register.
    Register(Register),
    /// Direct stack slot access (used for spill/fill or stack-allocated temps).
    Stack(StackIndex),
    /// Immediate scalar literal encoded inline.
    Immediate(ScalarValue),
    /// Handle into the shared constant pool.
    Constant(ConstantId),
}

/// Arithmetic and logical binary operators recognised by the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
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
}

/// Supported unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Marker instructions describing loop boundaries for structured control-flow.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopInstruction {
    /// Enter a loop (records the header block and syntactic depth).
    Enter { header: BlockId, depth: u16 },
    /// Continue statement targeting the header block.
    Continue { header: BlockId },
    /// Break statement targeting the exit block.
    Break { exit: BlockId },
    /// Exit marker signalling loop teardown.
    Exit { exit: BlockId },
}

/// High-level container operations that survive into the IR.
#[derive(Debug, Clone, PartialEq)]
pub enum ContainerOp {
    ListNew {
        target: Register,
        initial_arity: u16,
    },
    ListPush {
        list: Register,
        value: Operand,
    },
    ListPop {
        list: Register,
        target: Register,
    },
    MapInsert {
        map: Register,
        key: Operand,
        value: Operand,
    },
    MapGet {
        map: Register,
        key: Operand,
        target: Register,
    },
    SetInsert {
        set: Register,
        value: Operand,
    },
    Length {
        target: Register,
        container: Operand,
    },
}

/// Call target describing whether the call is direct or computed.
#[derive(Debug, Clone, PartialEq)]
pub enum CallSite {
    /// Direct call to another function within the module by name.
    Direct(String),
    /// Indirect call through a register/constant (first-class functions).
    Indirect(Operand),
    /// Built-in or host intrinsic.
    Intrinsic(String),
}

/// IR instruction set.
#[derive(Debug, Clone, PartialEq)]
pub enum IrInstruction {
    /// Load an immediate literal into a register.
    LoadImmediate {
        target: Register,
        value: ScalarValue,
    },
    /// Load a constant pool entry into a register.
    LoadConst {
        target: Register,
        constant: ConstantId,
    },
    /// Move/copy from one operand into a register (SSA phi lowering).
    Move { target: Register, source: Operand },
    /// Push an operand onto the evaluation stack.
    Push { source: Operand },
    /// Pop the top of stack, optionally discarding the value.
    Pop,
    /// Binary arithmetic/logic operation producing a register result.
    Binary {
        op: BinaryOp,
        lhs: Operand,
        rhs: Operand,
        target: Register,
    },
    /// Unary operation producing a register result.
    Unary {
        op: UnaryOp,
        operand: Operand,
        target: Register,
    },
    /// Unconditional branch to another block.
    Jump { target: BlockId },
    /// Conditional branch.
    Branch {
        condition: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    /// Loop structure markers (used for lowering structured loops).
    Loop(LoopInstruction),
    /// Call into another function/intrinsic.
    Call {
        site: CallSite,
        args: Vec<Operand>,
        result: Option<Register>,
    },
    /// Container-specific operation.
    Container { op: ContainerOp },
    /// Emit a runtime contract check.
    Contract {
        kind: ContractKind,
        annotation: ResourceAnnotation,
    },
    /// Return from the current function.
    Return { value: Option<Operand> },
}
