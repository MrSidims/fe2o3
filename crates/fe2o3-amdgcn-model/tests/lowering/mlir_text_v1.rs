//! Engineering-only MLIR `llvm`/`rocdl` emitter: structural, differential, and fail-closed checks
//! over the single-kernel lowering corpus.

use super::*;
use crate::mlir_differential_v1::{
    BOTH_TARGETS, MlirTarget, assert_equivalent, assert_fails_closed, exact,
};
use fe2o3_amdgcn_model::{
    lower_compiler_module_to_gfx942_xnack_minus_mlir_v1,
    lower_compiler_module_to_gfx950_xnack_minus_mlir_v1,
};

fn corpus() -> Vec<(&'static str, Module)> {
    vec![
        ("fill", fill_module()),
        ("private_pointer_slot", private_pointer_slot_module()),
        ("phi_loop", phi_loop_module()),
        ("vecadd", vecadd_module()),
        ("g4_synchronization", g4_synchronization_module()),
        ("scoped_atomics", scoped_atomics_module()),
        (
            "direct_global_pointer_atomic",
            direct_global_pointer_atomic_module(),
        ),
    ]
}

fn fill_with_entry_terminator(terminator: Terminator) -> Module {
    let mut module = fill_module();
    module.functions[0].body.as_mut().unwrap().blocks[0].terminator = Some(terminator);
    module
}

fn switch_modules() -> Vec<(&'static str, Module)> {
    let case = |value: u64, target: u32| SwitchCase {
        value,
        target: BlockId(target),
        arguments: vec![],
    };
    let integer_case = |value: u64, target: u32| IntegerSwitchCase {
        value: Constant::Index(value),
        target: BlockId(target),
        arguments: vec![],
    };
    vec![
        (
            "raw_switch",
            fill_with_entry_terminator(Terminator::Switch {
                selector: ValueId(2),
                cases: vec![case(7, 1), case(u64::MAX, 1), case(1 << 63, 2), case(0, 1)],
                default_target: BlockId(2),
                default_arguments: vec![],
            }),
        ),
        (
            "empty_raw_switch",
            fill_with_entry_terminator(Terminator::Switch {
                selector: ValueId(2),
                cases: vec![],
                default_target: BlockId(1),
                default_arguments: vec![],
            }),
        ),
        (
            "integer_switch",
            fill_with_entry_terminator(Terminator::IntegerSwitch {
                selector: ValueId(2),
                cases: vec![
                    integer_case(42, 1),
                    integer_case(1 << 63, 2),
                    integer_case(u64::MAX, 1),
                ],
                default_target: BlockId(2),
                default_arguments: vec![],
            }),
        ),
    ]
}

fn unreachable_block_module() -> Module {
    let mut module = fill_module();
    let mut dead = BasicBlock::new(BlockId(3));
    dead.terminator = Some(Terminator::Unreachable);
    module.functions[0].body.as_mut().unwrap().blocks.push(dead);
    module
}

#[test]
fn mlir_v1_single_kernel_corpus_matches_the_textual_emitter() {
    for target in BOTH_TARGETS {
        for (name, module) in corpus()
            .into_iter()
            .chain(switch_modules())
            .chain([("unreachable_block", unreachable_block_module())])
        {
            assert_equivalent(name, &exact(module, target), target);
        }
    }
}

#[test]
fn mlir_v1_switch_cases_are_signed_and_empty_lists_are_explicit() {
    for (name, module) in switch_modules() {
        let mlir = assert_equivalent(name, &exact(module, MlirTarget::Gfx942), MlirTarget::Gfx942);
        let words = mlir.split_whitespace().collect::<Vec<_>>().join(" ");
        let expected = match name {
            "raw_switch" => {
                "llvm.switch %v2 : i64, ^bb2 [ 7: ^bb1, -1: ^bb1, -9223372036854775808: ^bb2, 0: ^bb1 ]"
            }
            "empty_raw_switch" => "llvm.switch %v2 : i64, ^bb1 []",
            "integer_switch" => {
                "llvm.switch %v2 : i64, ^bb2 [ 42: ^bb1, -9223372036854775808: ^bb2, -1: ^bb1 ]"
            }
            _ => unreachable!(),
        };
        assert!(words.contains(expected), "{name}: {mlir}");
    }
}

#[test]
fn mlir_v1_emits_no_module_flags_or_named_metadata() {
    for (name, module) in corpus() {
        let mlir = assert_equivalent(name, &exact(module, MlirTarget::Gfx942), MlirTarget::Gfx942);
        assert!(!mlir.contains("llvm.module_flags"), "{name}");
        assert!(!mlir.contains("llvm.named_metadata"), "{name}");
        assert!(
            !mlir.contains("amdgpu.xnack"),
            "{name}: the worker owns the xnack flag"
        );
    }
}

#[test]
fn mlir_v1_is_deterministic_and_target_specific() {
    for (name, module) in corpus() {
        let gfx942 = exact(module.clone(), MlirTarget::Gfx942);
        let gfx950 = exact(module, MlirTarget::Gfx950);
        let first = lower_compiler_module_to_gfx942_xnack_minus_mlir_v1(&gfx942).unwrap();
        assert_eq!(
            first,
            lower_compiler_module_to_gfx942_xnack_minus_mlir_v1(&gfx942).unwrap(),
            "{name}"
        );
        let other = lower_compiler_module_to_gfx950_xnack_minus_mlir_v1(&gfx950).unwrap();
        assert!(first.contains("target_cpu = \"gfx942\""), "{name}");
        assert!(other.contains("target_cpu = \"gfx950\""), "{name}");
        assert!(!other.contains("gfx942"), "{name}");
    }
}

#[test]
fn mlir_v1_rejects_targets_it_was_not_bound_to() {
    let gfx950 = exact(fill_module(), MlirTarget::Gfx950);
    assert!(lower_compiler_module_to_gfx942_xnack_minus_mlir_v1(&gfx950).is_err());
    let gfx942 = exact(fill_module(), MlirTarget::Gfx942);
    assert!(lower_compiler_module_to_gfx950_xnack_minus_mlir_v1(&gfx942).is_err());
    assert!(lower_compiler_module_to_gfx942_xnack_minus_mlir_v1(&fill_module()).is_err());
}

#[test]
fn mlir_v1_fails_closed_outside_its_reviewed_subset() {
    for target in BOTH_TARGETS {
        assert_fails_closed("sqrt", &exact(sqrt_module(), target), target, "float");
        assert_fails_closed(
            "wave64",
            &exact(wave_module(WaveWidth::Wave64), target),
            target,
            "operation kind Wave",
        );
    }
    assert_fails_closed(
        "gfx950_bf16_mfma",
        &exact(gfx950_bf16_mfma_module(), MlirTarget::Gfx950),
        MlirTarget::Gfx950,
        "operation kind Matrix",
    );
}
