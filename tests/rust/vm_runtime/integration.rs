use praxis_vm_runtime::bytecode::{Opcode, Register, ScalarValue};
use praxis_vm_runtime::compiler::lower_module;
use praxis_vm_runtime::interpreter::Interpreter;
use praxis_vm_runtime::ir::{
    BinaryOp, BlockId, CallSite, ContainerOp, IrBlock, IrFunction, IrInstruction, IrLocal,
    IrModule, IrParameter, Operand,
};

fn build_sample_module() -> IrModule {
    let mut module = IrModule::new();

    let mut double_block = IrBlock::new(BlockId(0));
    double_block.push(IrInstruction::Binary {
        op: BinaryOp::Add,
        lhs: Operand::Register(Register(0)),
        rhs: Operand::Register(Register(0)),
        target: Register(1),
    });
    double_block.push(IrInstruction::Return {
        value: Some(Operand::Register(Register(1))),
    });

    let double_fn = IrFunction {
        name: "double".to_string(),
        params: vec![IrParameter {
            name: "value".to_string(),
            register: Register(0),
        }],
        locals: vec![IrLocal {
            name: "acc".to_string(),
            register: Register(1),
        }],
        entry: BlockId(0),
        blocks: vec![double_block],
        contracts: Vec::new(),
        stack_hint: Some(1),
    };
    module.add_function(double_fn);

    let mut main_block = IrBlock::new(BlockId(0));
    main_block.push(IrInstruction::Container {
        op: ContainerOp::ListNew {
            target: Register(0),
            initial_arity: 0,
        },
    });
    for (reg, immediate) in [(1u16, 1i64), (2u16, 2i64), (3u16, 3i64)] {
        main_block.push(IrInstruction::LoadImmediate {
            target: Register(reg),
            value: ScalarValue::Int(immediate),
        });
        main_block.push(IrInstruction::Container {
            op: ContainerOp::ListPush {
                list: Register(0),
                value: Operand::Register(Register(reg)),
            },
        });
    }
    main_block.push(IrInstruction::Call {
        site: CallSite::Intrinsic("sum".to_string()),
        args: vec![Operand::Register(Register(0))],
        result: Some(Register(4)),
    });
    main_block.push(IrInstruction::Call {
        site: CallSite::Direct("double".to_string()),
        args: vec![Operand::Register(Register(4))],
        result: Some(Register(5)),
    });
    main_block.push(IrInstruction::Return {
        value: Some(Operand::Register(Register(5))),
    });

    let locals: Vec<IrLocal> = (0u16..=5u16)
        .map(|index| IrLocal {
            name: format!("r{index}"),
            register: Register(index),
        })
        .collect();

    let main_fn = IrFunction {
        name: "main".to_string(),
        params: Vec::new(),
        locals,
        entry: BlockId(0),
        blocks: vec![main_block],
        contracts: Vec::new(),
        stack_hint: Some(4),
    };
    module.add_function(main_fn);

    module
}

#[test]
fn executes_lowered_module_across_builtins_and_direct_calls() {
    let module = build_sample_module();
    let bytecode = lower_module(&module).expect("lowering should succeed");
    let interpreter = Interpreter::new(bytecode);

    let outcome = interpreter
        .execute("main", Vec::new())
        .expect("execution should succeed");
    let value = outcome.return_value.expect("main must return a value");
    assert_eq!(value, ScalarValue::Int(12).into());

    let trace = outcome.trace;
    assert!(trace.metrics.instructions >= 1);
    assert!(trace.metrics.unique_opcodes.contains(&Opcode::Call));
    assert!(trace
        .coverage
        .functions
        .get("main")
        .map(|pcs| !pcs.is_empty())
        .unwrap_or(false));
    assert!(trace.coverage.functions.contains_key("double"));
    assert!(trace.metrics.heap.allocations >= 1);
}
