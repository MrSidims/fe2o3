//! Engineering-only MLIR `llvm`/`rocdl` dialect text emitter for compiler modules (Plan A).
//!
//! This renderer consumes the exact same preflighted [`FunctionLowerer`] state as the textual
//! LLVM IR renderer and mirrors its instruction selection one-for-one, so a module translated
//! by `mlir-translate --mlir-to-llvmir` (or the worker's `MlirLlvmText` input) is semantically
//! equivalent to the textual LLVM output. It is deliberately narrower than the textual
//! renderer: every feature outside the reviewed subset fails closed before any text escapes.
//!
//! It is not a production or authority path. No receipt, digest, or lineage binds its bytes.

use super::*;

/// Selects the renderer that consumes a fully preflighted compiler module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompilerModuleEmitterV1 {
    /// The reviewed textual LLVM IR renderer.
    LlvmIr,
    /// The engineering-only MLIR `llvm`/`rocdl` dialect renderer.
    MlirLlvmV1,
}

/// Lowers a complete compiler module for gfx942:xnack- as MLIR `llvm`/`rocdl` dialect text.
///
/// Engineering-only: the output is meant for `mlir-translate --mlir-to-llvmir` or the worker's
/// MLIR input kind. Modules that need float-support helpers, diagnostics, memory intrinsics,
/// matrix, wave, inline-assembly, ordered, or semantic-anchor lowering are rejected.
pub fn lower_compiler_module_to_gfx942_xnack_minus_mlir_v1(
    module: &Module,
) -> Result<String, LoweringErrors> {
    lower_compiler_module_with_emitter_v1(
        module,
        LoweringTarget::Gfx942XnackMinusV1,
        None,
        None,
        true,
        None,
        None,
        None,
        None,
        false,
        CompilerModuleEmitterV1::MlirLlvmV1,
    )
}

/// Gfx950 counterpart of [`lower_compiler_module_to_gfx942_xnack_minus_mlir_v1`].
pub fn lower_compiler_module_to_gfx950_xnack_minus_mlir_v1(
    module: &Module,
) -> Result<String, LoweringErrors> {
    lower_compiler_module_with_emitter_v1(
        module,
        LoweringTarget::Gfx950XnackMinusV1,
        None,
        None,
        true,
        None,
        None,
        None,
        None,
        false,
        CompilerModuleEmitterV1::MlirLlvmV1,
    )
}

const DENORMAL_FPENV_IEEE: &str = "#llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = ieee, float_input_mode = ieee>";
const FP_PASSTHROUGH: &str = "[\"unsafe-fp-math\", \"false\"], [\"no-infs-fp-math\", \"false\"], [\"no-nans-fp-math\", \"false\"], [\"no-signed-zeros-fp-math\", \"false\"], [\"approx-func-fp-math\", \"false\"], [\"fp-contract\", \"off\"]";
const GLOBAL_PTR: &str = "!llvm.ptr<1>";
const CONSTANT_PTR: &str = "!llvm.ptr<4>";
const INDEX_TY: &str = "i64";

fn unsupported(location: LoweringLocation, message: impl Into<String>) -> LoweringErrors {
    LoweringErrors::one(
        location,
        LoweringDiagnosticCode::UnsupportedOperation,
        format!("MLIR LLVM V1 emitter: {}", message.into()),
    )
}

pub(super) fn emit_compiler_module_mlir_v1(
    module: &Module,
    kernels: &[FunctionLowerer<'_>],
    helpers: &[FunctionLowerer<'_>],
    declarations: &[&Function],
    target: LoweringTarget,
    semantic_anchor_identity: Option<SemanticAnchorInputV1<'_>>,
) -> Result<String, LoweringErrors> {
    let module_location = || LoweringLocation::module(module);
    let target_cpu = match target {
        LoweringTarget::Gfx942XnackMinusV1 => "gfx942",
        LoweringTarget::Gfx950XnackMinusV1 => "gfx950",
        LoweringTarget::Baseline | LoweringTarget::Gfx942StrictFloatV1 => {
            return Err(unsupported(
                module_location(),
                "only the exact gfx942/gfx950 xnack- compiler-module targets are admitted",
            ));
        }
    };
    if semantic_anchor_identity.is_some() {
        return Err(unsupported(
            module_location(),
            "semantic anchors are not representable",
        ));
    }
    for lowerer in kernels.iter().chain(helpers) {
        if !matches!(
            lowerer.semantic_anchor_emission,
            SemanticAnchorEmissionV1::Disabled
        ) || lowerer.checked_load_grouping_v1
            || lowerer.ordered_region_v16
            || lowerer.ordered_program_v17
            || lowerer.ordered_debug_line_v17.is_some()
            || lowerer.ordered_composition_v1.is_some()
            || lowerer.native_bf16_helper.is_some()
        {
            return Err(unsupported(
                lowerer.function_location(),
                "anchor, checked-load grouping, ordered, debug-line, and native-bf16 contexts are not representable",
            ));
        }
        if lowerer.target != target {
            return Err(unsupported(
                lowerer.function_location(),
                "function target differs from the module target",
            ));
        }
    }
    if !FloatRequirements::collect(kernels.iter().chain(helpers))?.is_empty() {
        return Err(unsupported(
            module_location(),
            "float-support helpers (narrow floats, OCML math) are not supported yet",
        ));
    }
    if !DiagnosticRequirements::collect(kernels.iter().chain(helpers))?.is_empty() {
        return Err(unsupported(
            module_location(),
            "AMDGPU diagnostics are not supported",
        ));
    }
    if !collect_memcpy_declarations(kernels.iter().chain(helpers))?.is_empty() {
        return Err(unsupported(
            module_location(),
            "memory intrinsics are not supported",
        ));
    }
    let data_layout = target
        .data_layout()
        .ok_or_else(|| unsupported(module_location(), "target has no reviewed data layout"))?;

    let mut output = CapacityLimitedText::try_new(module, MAX_COMPILER_MODULE_TEXT_BYTES)?;
    writeln!(
        output,
        "module attributes {{llvm.target_triple = \"{AMDGPU_TRIPLE}\", llvm.data_layout = \"{data_layout}\"}} {{"
    )
    .unwrap();
    for lowerer in kernels {
        emit_workgroup_memory_globals(&mut output, lowerer)?;
    }
    for function in declarations {
        emit_external_declaration(&mut output, module, function)?;
    }
    for lowerer in kernels.iter().chain(helpers) {
        FunctionEmitter::new(lowerer, target_cpu).emit(&mut output)?;
    }
    writeln!(output, "}}").unwrap();
    output.finish(module)
}

fn mlir_scalar(scalar: ScalarType) -> &'static str {
    match scalar {
        ScalarType::Bool => "i1",
        ScalarType::I8 | ScalarType::U8 => "i8",
        ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::Bf16 => "i16",
        ScalarType::I32 | ScalarType::U32 => "i32",
        ScalarType::I64 | ScalarType::U64 | ScalarType::Index => "i64",
        ScalarType::I128 | ScalarType::U128 => "i128",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
    }
}

fn mlir_pointer(address_space: KernelAddressSpace) -> Option<&'static str> {
    match address_space {
        KernelAddressSpace::Global => Some(GLOBAL_PTR),
        KernelAddressSpace::Workgroup => Some("!llvm.ptr<3>"),
        KernelAddressSpace::Private => Some("!llvm.ptr<5>"),
        _ => None,
    }
}

/// Maps a first-class KIR value type to its MLIR LLVM dialect spelling.
fn mlir_type(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Scalar(scalar) => Some(mlir_scalar(*scalar)),
        Type::Pointer(pointer) => mlir_pointer(pointer.address_space),
        _ => None,
    }
}

fn mlir_type_at(ty: &Type, location: &LoweringLocation) -> Result<&'static str, LoweringErrors> {
    mlir_type(ty).ok_or_else(|| {
        LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedType,
            format!("MLIR LLVM V1 emitter: type {ty:?} has no MLIR LLVM dialect mapping"),
        )
    })
}

fn integer_width(mlir_ty: &str) -> Option<u32> {
    mlir_ty.strip_prefix('i')?.parse().ok()
}

fn struct_type(types: &[&str]) -> String {
    format!("!llvm.struct<({})>", types.join(", "))
}

fn check_alignment(alignment: u32, location: &LoweringLocation) -> Result<(), LoweringErrors> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(unsupported(
            location.clone(),
            format!("alignment {alignment} is not a nonzero power of two"),
        ));
    }
    Ok(())
}

fn check_symbol(symbol: &str, location: &LoweringLocation) -> Result<(), LoweringErrors> {
    if !is_safe_symbol(symbol) {
        return Err(unsupported(
            location.clone(),
            format!("symbol {symbol:?} is not a safe MLIR symbol"),
        ));
    }
    Ok(())
}

/// The literal attribute and type of a hoisted constant.
fn mlir_constant(constant: &Constant) -> (String, &'static str) {
    match constant {
        Constant::Bool(value) => (value.to_string(), "i1"),
        Constant::I8(value) => (format!("{value} : i8"), "i8"),
        Constant::U8(value) => (format!("{} : i8", *value as i8), "i8"),
        Constant::I16(value) => (format!("{value} : i16"), "i16"),
        Constant::U16(value) => (format!("{} : i16", *value as i16), "i16"),
        Constant::F16Bits(value) | Constant::Bf16Bits(value) => {
            (format!("{} : i16", *value as i16), "i16")
        }
        Constant::I32(value) => (format!("{value} : i32"), "i32"),
        Constant::U32(value) => (format!("{} : i32", *value as i32), "i32"),
        Constant::I64(value) => (format!("{value} : i64"), "i64"),
        Constant::U64(value) | Constant::Index(value) => {
            (format!("{} : i64", *value as i64), "i64")
        }
        Constant::F32Bits(bits) => (format!("0x{bits:08X} : f32"), "f32"),
        Constant::F64Bits(bits) => (format!("0x{bits:016X} : f64"), "f64"),
    }
}

/// Signed reinterpretation of an integer switch constant at its own width.
fn integer_switch_case_value(constant: &Constant) -> Option<i64> {
    Some(match constant {
        Constant::Bool(value) => -i64::from(*value),
        Constant::I8(value) => i64::from(*value),
        Constant::U8(value) => i64::from(*value as i8),
        Constant::I16(value) => i64::from(*value),
        Constant::U16(value) => i64::from(*value as i16),
        Constant::I32(value) => i64::from(*value),
        Constant::U32(value) => i64::from(*value as i32),
        Constant::I64(value) => *value,
        Constant::U64(value) | Constant::Index(value) => *value as i64,
        Constant::F16Bits(_)
        | Constant::Bf16Bits(_)
        | Constant::F32Bits(_)
        | Constant::F64Bits(_) => return None,
    })
}

/// Encodes a raw `u64` switch case for an `iW` selector as MLIR's signed case literal.
fn raw_switch_case_value(value: u64, width: u32) -> Option<i64> {
    match width {
        1..=63 => {
            if value >> width != 0 {
                return None;
            }
            let shift = 64 - width;
            Some(((value << shift) as i64) >> shift)
        }
        64 => Some(value as i64),
        128 => i64::try_from(value).ok(),
        _ => None,
    }
}

fn mlir_binary_opcode(op: BinaryOp, ty: &Type) -> Option<&'static str> {
    let scalar = ty.as_scalar()?;
    let floating = matches!(scalar, ScalarType::F32 | ScalarType::F64);
    let signed = scalar.is_signed_integer();
    Some(match (op, floating) {
        (BinaryOp::Add, false) => "add",
        (BinaryOp::Subtract, false) => "sub",
        (BinaryOp::Multiply, false) => "mul",
        (BinaryOp::Divide, false) if signed => "sdiv",
        (BinaryOp::Divide, false) => "udiv",
        (BinaryOp::Remainder, false) if signed => "srem",
        (BinaryOp::Remainder, false) => "urem",
        (BinaryOp::BitAnd, false) => "and",
        (BinaryOp::BitOr, false) => "or",
        (BinaryOp::BitXor, false) => "xor",
        (BinaryOp::ShiftLeft, false) => "shl",
        (BinaryOp::ShiftRight, false) if signed => "ashr",
        (BinaryOp::ShiftRight, false) => "lshr",
        (BinaryOp::Add, true) => "fadd",
        (BinaryOp::Subtract, true) => "fsub",
        (BinaryOp::Multiply, true) => "fmul",
        (BinaryOp::Divide, true) => "fdiv",
        _ => return None,
    })
}

fn mlir_atomic_rmw_opcode(kind: AtomicKind, scalar: ScalarType) -> Option<&'static str> {
    let integer = scalar.is_integer();
    Some(match kind {
        AtomicKind::Exchange => "xchg",
        AtomicKind::Add if integer => "add",
        AtomicKind::Subtract if integer => "sub",
        AtomicKind::Min if integer && scalar.is_signed_integer() => "min",
        AtomicKind::Min if integer => "umin",
        AtomicKind::Max if integer && scalar.is_signed_integer() => "max",
        AtomicKind::Max if integer => "umax",
        AtomicKind::BitAnd if integer => "_and",
        AtomicKind::BitOr if integer => "_or",
        AtomicKind::BitXor if integer => "_xor",
        _ => return None,
    })
}

fn mlir_atomic_sync_scope(scope: SynchronizationScope) -> Option<&'static str> {
    match scope {
        SynchronizationScope::Workgroup => Some(" syncscope(\"workgroup\")"),
        SynchronizationScope::Device => Some(" syncscope(\"agent\")"),
        SynchronizationScope::System => Some(""),
        SynchronizationScope::Invocation | SynchronizationScope::Subgroup => None,
    }
}

fn mlir_fence_sync_scope(scope: SynchronizationScope) -> Option<&'static str> {
    match scope {
        SynchronizationScope::Subgroup => Some(" syncscope(\"wavefront\")"),
        SynchronizationScope::Workgroup => Some(" syncscope(\"workgroup\")"),
        SynchronizationScope::Device => Some(" syncscope(\"agent\")"),
        SynchronizationScope::System => Some(""),
        SynchronizationScope::Invocation => None,
    }
}

fn mlir_wave_target_features(width: WaveWidth) -> &'static str {
    match width {
        WaveWidth::Wave32 => {
            "#llvm.target_features<[\"+wavefrontsize32\", \"-wavefrontsize64\", \"-xnack\"]>"
        }
        WaveWidth::Wave64 => {
            "#llvm.target_features<[\"-wavefrontsize32\", \"+wavefrontsize64\", \"-xnack\"]>"
        }
    }
}

fn emit_workgroup_memory_globals(
    output: &mut dyn fmt::Write,
    lowerer: &FunctionLowerer<'_>,
) -> Result<(), LoweringErrors> {
    let body = lowerer.body("function body is missing during MLIR LDS declaration scan")?;
    for block in &body.blocks {
        for (operation_index, operation) in block.operations.iter().enumerate() {
            let OperationKind::WorkgroupMemory(memory) = &operation.kind else {
                continue;
            };
            let location = lowerer.operation_location(block.id, operation_index);
            let kernel = lowerer
                .kernel
                .ok_or_else(|| unsupported(location.clone(), "LDS requires a kernel"))?;
            let [result] = operation.results.as_slice() else {
                return Err(unsupported(
                    location,
                    "LDS declaration must define one result",
                ));
            };
            let symbol = lds_symbol(kernel, result.id);
            let symbol = symbol.strip_prefix('@').unwrap_or(&symbol);
            check_symbol(symbol, &location)?;
            check_alignment(memory.alignment, &location)?;
            let element = mlir_type_at(&memory.element, &location)?;
            match memory.extent {
                WorkgroupMemoryExtent::Static(elements) => {
                    let array = format!("!llvm.array<{elements} x {element}>");
                    writeln!(
                        output,
                        "  llvm.mlir.global internal @{symbol}() {{addr_space = 3 : i32, alignment = {} : i64}} : {array} {{",
                        memory.alignment
                    )
                    .unwrap();
                    writeln!(output, "    %0 = llvm.mlir.undef : {array}").unwrap();
                    writeln!(output, "    llvm.return %0 : {array}").unwrap();
                    writeln!(output, "  }}").unwrap();
                }
                WorkgroupMemoryExtent::Dynamic | WorkgroupMemoryExtent::DynamicAtLeast(_) => {
                    writeln!(
                        output,
                        "  llvm.mlir.global external @{symbol}() {{addr_space = 3 : i32, alignment = {} : i64}} : !llvm.array<0 x {element}>",
                        memory.alignment
                    )
                    .unwrap();
                }
            }
        }
    }
    Ok(())
}

fn emit_external_declaration(
    output: &mut dyn fmt::Write,
    module: &Module,
    function: &Function,
) -> Result<(), LoweringErrors> {
    let location = LoweringLocation::device_function(module, function);
    let symbol = function.id.as_str();
    check_symbol(symbol, &location)?;
    let parameters = function
        .signature
        .parameters
        .iter()
        .map(|ty| mlir_type_at(ty, &location))
        .collect::<Result<Vec<_>, _>>()?;
    let result = match function.signature.results.as_slice() {
        [] => String::new(),
        [result] => format!(" -> {}", mlir_type_at(result, &location)?),
        _ => {
            return Err(unsupported(
                location,
                "external declarations with multiple results are not supported",
            ));
        }
    };
    writeln!(
        output,
        "  llvm.func @{symbol}({}){result}",
        parameters.join(", ")
    )
    .unwrap();
    Ok(())
}

struct FunctionEmitter<'e, 'a> {
    lowerer: &'e FunctionLowerer<'a>,
    target_cpu: &'static str,
    constants: BTreeSet<ValueId>,
    next_immediate: u64,
}

impl<'e, 'a> FunctionEmitter<'e, 'a> {
    fn new(lowerer: &'e FunctionLowerer<'a>, target_cpu: &'static str) -> Self {
        Self {
            lowerer,
            target_cpu,
            constants: BTreeSet::new(),
            next_immediate: 0,
        }
    }

    fn function_location(&self) -> LoweringLocation {
        self.lowerer.function_location()
    }

    fn operation_location(&self, block: BlockId, operation: usize) -> LoweringLocation {
        self.lowerer.operation_location(block, operation)
    }

    fn block_location(&self, block: BlockId) -> LoweringLocation {
        self.lowerer.block_location(block)
    }

    /// Emits a fresh integer immediate right before its use and returns its SSA name.
    fn immediate(
        &mut self,
        output: &mut dyn fmt::Write,
        value: impl fmt::Display,
        ty: &str,
        location: &LoweringLocation,
    ) -> Result<String, LoweringErrors> {
        if integer_width(ty).is_none() {
            return Err(unsupported(
                location.clone(),
                format!("immediate type {ty} is not an integer"),
            ));
        }
        let name = format!("%imm{}", self.next_immediate);
        self.next_immediate += 1;
        if ty == "i1" {
            return Err(unsupported(
                location.clone(),
                "i1 immediates must use the boolean form",
            ));
        }
        writeln!(
            output,
            "    {name} = llvm.mlir.constant({value} : {ty}) : {ty}"
        )
        .unwrap();
        Ok(name)
    }

    fn immediate_true(&mut self, output: &mut dyn fmt::Write) -> String {
        let name = format!("%imm{}", self.next_immediate);
        self.next_immediate += 1;
        writeln!(output, "    {name} = llvm.mlir.constant(true) : i1").unwrap();
        name
    }

    /// Resolves a scalar or pointer operand to its MLIR SSA name and KIR type.
    fn operand(
        &self,
        value: ValueId,
        location: &LoweringLocation,
    ) -> Result<(String, &'e Type), LoweringErrors> {
        match self.lowerer.bindings.get(&value) {
            Some(ValueBinding::Value { llvm_name, ty, .. }) => {
                if llvm_name.starts_with('%') {
                    Ok((llvm_name.clone(), ty))
                } else if self.constants.contains(&value) {
                    Ok((value_name(value), ty))
                } else {
                    Err(unsupported(
                        location.clone(),
                        format!(
                            "value {value} is bound to a non-SSA literal that is not a hoisted constant"
                        ),
                    ))
                }
            }
            Some(ValueBinding::Slice { .. }) => Err(unsupported(
                location.clone(),
                format!("slice value {value} used where a scalar or pointer is required"),
            )),
            None => Err(unsupported(
                location.clone(),
                format!("value {value} has no binding"),
            )),
        }
    }

    fn typed_operand(
        &self,
        value: ValueId,
        location: &LoweringLocation,
    ) -> Result<(String, &'e Type, &'static str), LoweringErrors> {
        let (name, ty) = self.operand(value, location)?;
        let mlir = mlir_type_at(ty, location)?;
        Ok((name, ty, mlir))
    }

    fn slice_operand(
        &self,
        value: ValueId,
        location: &LoweringLocation,
    ) -> Result<(&'e str, &'e str), LoweringErrors> {
        match self.lowerer.bindings.get(&value) {
            Some(ValueBinding::Slice {
                data_name,
                length_name,
                ..
            }) if data_name.starts_with('%') && length_name.starts_with('%') => {
                Ok((data_name, length_name))
            }
            _ => Err(unsupported(
                location.clone(),
                format!("value {value} is not a bound slice"),
            )),
        }
    }

    /// Appends the expanded `(name, type)` pairs of a value passed across a call or edge.
    fn push_flattened(
        &self,
        value: ValueId,
        location: &LoweringLocation,
        names: &mut Vec<String>,
        types: &mut Vec<&'static str>,
    ) -> Result<(), LoweringErrors> {
        match self.lowerer.bindings.get(&value) {
            Some(ValueBinding::Slice { .. }) => {
                let (data, length) = self.slice_operand(value, location)?;
                names.push(data.to_owned());
                names.push(length.to_owned());
                types.push(GLOBAL_PTR);
                types.push(INDEX_TY);
            }
            _ => {
                let (name, _, mlir) = self.typed_operand(value, location)?;
                names.push(name);
                types.push(mlir);
            }
        }
        Ok(())
    }

    /// Renders a binding as a block or function argument declaration.
    fn push_argument_declaration(
        &self,
        value: ValueId,
        location: &LoweringLocation,
        with_location: bool,
        kernel_entry: bool,
        arguments: &mut Vec<String>,
    ) -> Result<(), LoweringErrors> {
        let render = |name: &str, ty: &str| {
            if with_location {
                format!("{name}: {ty} loc(\"{}\")", &name[1..])
            } else {
                format!("{name}: {ty}")
            }
        };
        match self.lowerer.bindings.get(&value) {
            Some(ValueBinding::Value { llvm_name, ty, .. }) if llvm_name.starts_with('%') => {
                if kernel_entry
                    && matches!(ty, Type::Pointer(pointer) if pointer.address_space != KernelAddressSpace::Global)
                {
                    return Err(unsupported(
                        location.clone(),
                        "kernel pointer parameters must be global",
                    ));
                }
                arguments.push(render(llvm_name, mlir_type_at(ty, location)?));
            }
            Some(ValueBinding::Slice { .. }) => {
                let (data, length) = self.slice_operand(value, location)?;
                arguments.push(render(data, GLOBAL_PTR));
                arguments.push(render(length, INDEX_TY));
            }
            _ => {
                return Err(unsupported(
                    location.clone(),
                    format!("argument {value} has no SSA binding"),
                ));
            }
        }
        Ok(())
    }

    fn signature_result(&self) -> Result<(String, Option<String>), LoweringErrors> {
        let location = self.function_location();
        let results = self
            .lowerer
            .function
            .signature
            .results
            .iter()
            .map(|ty| mlir_type_at(ty, &location))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(match results.as_slice() {
            [] => (String::new(), None),
            [result] => (format!(" -> {result}"), None),
            _ => {
                let ty = struct_type(&results);
                (format!(" -> {ty}"), Some(ty))
            }
        })
    }

    fn emit(mut self, output: &mut dyn fmt::Write) -> Result<(), LoweringErrors> {
        let lowerer = self.lowerer;
        let location = self.function_location();
        check_symbol(lowerer.symbol, &location)?;
        let body = lowerer.body("function body is missing during MLIR emission")?;
        if body.parameters.len() != lowerer.function.signature.parameters.len() {
            return Err(unsupported(
                location,
                "function body parameters differ from its signature",
            ));
        }
        let kernel_entry = lowerer.kernel.is_some();
        let mut parameters = Vec::new();
        for value in &body.parameters {
            self.push_argument_declaration(*value, &location, true, kernel_entry, &mut parameters)?;
        }
        let parameters = parameters.join(", ");
        let convergent = lowerer.wave64_helper_context_v1.is_some_and(|context| {
            context.function_is_convergent(lowerer.module, lowerer.function)
        });
        let mut attributes = vec!["no_unwind".to_owned()];
        if convergent {
            attributes.push("convergent".to_owned());
        }
        attributes.push(format!("target_cpu = \"{}\"", self.target_cpu));
        if let Some(width) = lowerer.wave_width {
            attributes.push(format!(
                "target_features = {}",
                mlir_wave_target_features(width)
            ));
        }
        attributes.push(format!("denormal_fpenv = {DENORMAL_FPENV_IEEE}"));
        if lowerer.kernel.is_some() {
            if !lowerer.function.signature.results.is_empty() {
                return Err(unsupported(
                    location,
                    "kernel entries must not return values",
                ));
            }
            let flat = lowerer
                .required_flat_workgroup_size("compiler-module kernel requires a workgroup size")?;
            let size = lowerer
                .required_workgroup_size("compiler-module kernel requires a workgroup size")?;
            let waves = lowerer.launch_bounds.map_or_else(String::new, |bounds| {
                debug_assert!(bounds.admits_flat_workgroup_size(flat));
                format!(
                    "[\"amdgpu-waves-per-eu\", \"{},{}\"], ",
                    bounds.minimum_waves_per_execution_unit(),
                    bounds.maximum_waves_per_execution_unit()
                )
            });
            attributes.push(format!(
                "passthrough = [[\"amdgpu-flat-work-group-size\", \"{flat},{flat}\"], {waves}{FP_PASSTHROUGH}]"
            ));
            attributes.push(format!(
                "reqd_work_group_size = array<i32: {}, {}, {}>",
                size.x, size.y, size.z
            ));
            writeln!(
                output,
                "  llvm.func amdgpu_kernelcc @{}({parameters}) attributes {{{}}} {{",
                lowerer.symbol,
                attributes.join(", ")
            )
            .unwrap();
        } else {
            let linkage = match lowerer.function.role {
                FunctionRole::InternalHelper => "internal ",
                FunctionRole::DeviceFfiExport => "",
                FunctionRole::KernelEntry | FunctionRole::ExternalImport => {
                    return Err(unsupported(
                        location,
                        "compiler-module helper definition has a non-definition role",
                    ));
                }
            };
            let (result, _) = self.signature_result()?;
            attributes.push(format!("passthrough = [{FP_PASSTHROUGH}]"));
            writeln!(
                output,
                "  llvm.func {linkage}@{}({parameters}){result} attributes {{{}}} {{",
                lowerer.symbol,
                attributes.join(", ")
            )
            .unwrap();
        }
        self.emit_body(output, body)?;
        writeln!(output, "  }}").unwrap();
        Ok(())
    }

    fn emit_body(
        &mut self,
        output: &mut dyn fmt::Write,
        body: &'e FunctionBody,
    ) -> Result<(), LoweringErrors> {
        let Some(entry) = body.blocks.first() else {
            return Err(unsupported(
                self.function_location(),
                "function body has no blocks",
            ));
        };
        let entry_has_predecessors = self
            .lowerer
            .control_flow
            .incoming_edges(entry.id)
            .is_none_or(|incoming| !incoming.is_empty());
        if !entry.parameters.is_empty() || entry_has_predecessors {
            return Err(unsupported(
                self.block_location(entry.id),
                "the entry block must have no parameters and no predecessors",
            ));
        }
        for block in &body.blocks {
            for (operation_index, operation) in block.operations.iter().enumerate() {
                let OperationKind::Constant(constant) = &operation.kind else {
                    continue;
                };
                let location = self.operation_location(block.id, operation_index);
                let [result] = operation.results.as_slice() else {
                    return Err(unsupported(location, "constants must define one result"));
                };
                let (literal, ty) = mlir_constant(constant);
                if mlir_type_at(&result.ty, &location)? != ty {
                    return Err(unsupported(
                        location,
                        "constant type differs from its result type",
                    ));
                }
                if !self.constants.insert(result.id) {
                    return Err(unsupported(location, "constant result is defined twice"));
                }
                if ty == "i1" {
                    writeln!(
                        output,
                        "    {} = llvm.mlir.constant({literal}) : i1",
                        value_name(result.id)
                    )
                    .unwrap();
                } else {
                    writeln!(
                        output,
                        "    {} = llvm.mlir.constant({literal}) : {ty}",
                        value_name(result.id)
                    )
                    .unwrap();
                }
            }
        }
        for (position, block) in body.blocks.iter().enumerate() {
            if position != 0 {
                let location = self.block_location(block.id);
                let mut parameters = Vec::new();
                for parameter in &block.parameters {
                    self.push_argument_declaration(
                        parameter.id,
                        &location,
                        false,
                        false,
                        &mut parameters,
                    )?;
                }
                if parameters.is_empty() {
                    writeln!(output, "  ^{}:", block_label(block.id)).unwrap();
                } else {
                    writeln!(
                        output,
                        "  ^{}({}):",
                        block_label(block.id),
                        parameters.join(", ")
                    )
                    .unwrap();
                }
            }
            for (operation_index, operation) in block.operations.iter().enumerate() {
                self.emit_operation(output, block.id, operation_index, operation)?;
            }
            let terminator = block.terminator.as_ref().ok_or_else(|| {
                LoweringErrors::one(
                    self.block_location(block.id),
                    LoweringDiagnosticCode::UnsupportedTerminator,
                    "basic block terminator is missing during MLIR emission",
                )
            })?;
            self.emit_terminator(output, block, terminator)?;
            self.emit_split_edges(output, block)?;
        }
        Ok(())
    }

    fn edge_index(
        &self,
        predecessor: BlockId,
        ordinal: usize,
        target: BlockId,
        location: &LoweringLocation,
    ) -> Result<usize, LoweringErrors> {
        let control_flow = &self.lowerer.control_flow;
        let outgoing = control_flow
            .outgoing_edges(predecessor)
            .ok_or_else(|| unsupported(location.clone(), "predecessor is not indexed"))?;
        let edge_index = outgoing
            .start
            .checked_add(ordinal)
            .filter(|edge_index| *edge_index < outgoing.end)
            .ok_or_else(|| unsupported(location.clone(), "edge ordinal is out of range"))?;
        if control_flow.edge_target(edge_index) != Some(target) {
            return Err(unsupported(
                location.clone(),
                "edge target differs from the control-flow index",
            ));
        }
        if edge_index >= self.lowerer.split_edges.len() {
            return Err(unsupported(location.clone(), "edge has no split decision"));
        }
        Ok(edge_index)
    }

    /// Renders `^bbT(args : types)` for an edge, checking arity against the target block.
    fn edge_destination(
        &self,
        edge_index: usize,
        target: BlockId,
        location: &LoweringLocation,
    ) -> Result<String, LoweringErrors> {
        let arguments = self
            .lowerer
            .control_flow
            .edge_arguments(self.lowerer.function, edge_index);
        let target_block = self.lowerer.block(target)?;
        if arguments.len() != target_block.parameters.len() {
            return Err(unsupported(
                location.clone(),
                "edge argument count differs from the target block parameters",
            ));
        }
        if arguments.is_empty() {
            return Ok(format!("^{}", block_label(target)));
        }
        let mut names = Vec::new();
        let mut types = Vec::new();
        for argument in arguments {
            self.push_flattened(*argument, location, &mut names, &mut types)?;
        }
        Ok(format!(
            "^{}({} : {})",
            block_label(target),
            names.join(", "),
            types.join(", ")
        ))
    }

    fn successor(
        &self,
        predecessor: BlockId,
        ordinal: usize,
        target: BlockId,
        location: &LoweringLocation,
    ) -> Result<String, LoweringErrors> {
        let edge_index = self.edge_index(predecessor, ordinal, target, location)?;
        if self.lowerer.split_edges[edge_index] {
            Ok(format!("^{}", edge_label(predecessor, ordinal, target)))
        } else {
            self.edge_destination(edge_index, target, location)
        }
    }

    fn emit_split_edges(
        &self,
        output: &mut dyn fmt::Write,
        block: &BasicBlock,
    ) -> Result<(), LoweringErrors> {
        let location = self.block_location(block.id);
        let outgoing = self
            .lowerer
            .control_flow
            .outgoing_edges(block.id)
            .ok_or_else(|| unsupported(location.clone(), "source block is not indexed"))?;
        for edge_index in outgoing {
            if !self
                .lowerer
                .split_edges
                .get(edge_index)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let edge = self
                .lowerer
                .control_flow
                .edge(edge_index)
                .ok_or_else(|| unsupported(location.clone(), "split edge is not indexed"))?;
            let target = self
                .lowerer
                .control_flow
                .edge_target(edge_index)
                .ok_or_else(|| unsupported(location.clone(), "split edge has no target"))?;
            writeln!(
                output,
                "  ^{}:",
                edge_label(block.id, edge.ordinal(), target)
            )
            .unwrap();
            writeln!(
                output,
                "    llvm.br {}",
                self.edge_destination(edge_index, target, &location)?
            )
            .unwrap();
        }
        Ok(())
    }

    fn emit_terminator(
        &mut self,
        output: &mut dyn fmt::Write,
        block: &BasicBlock,
        terminator: &Terminator,
    ) -> Result<(), LoweringErrors> {
        let predecessor = block.id;
        let location = self.block_location(predecessor);
        match terminator {
            Terminator::Branch { target, .. } => {
                writeln!(
                    output,
                    "    llvm.br {}",
                    self.successor(predecessor, 0, *target, &location)?
                )
                .unwrap();
            }
            Terminator::ConditionalBranch {
                condition,
                then_target,
                else_target,
                ..
            } => {
                let (condition, _, condition_ty) = self.typed_operand(*condition, &location)?;
                if condition_ty != "i1" {
                    return Err(unsupported(location, "branch condition must be i1"));
                }
                writeln!(
                    output,
                    "    llvm.cond_br {condition}, {}, {}",
                    self.successor(predecessor, 0, *then_target, &location)?,
                    self.successor(predecessor, 1, *else_target, &location)?
                )
                .unwrap();
            }
            Terminator::Switch {
                selector,
                cases,
                default_target,
                ..
            } => {
                let (selector, _, selector_ty) = self.typed_operand(*selector, &location)?;
                let width = integer_width(selector_ty).ok_or_else(|| {
                    unsupported(location.clone(), "switch selector must be an integer")
                })?;
                let mut rendered = Vec::with_capacity(cases.len());
                for (ordinal, case) in cases.iter().enumerate() {
                    let value = raw_switch_case_value(case.value, width).ok_or_else(|| {
                        unsupported(
                            location.clone(),
                            format!(
                                "switch case {} is not representable at i{width}",
                                case.value
                            ),
                        )
                    })?;
                    rendered.push(format!(
                        "{value}: {}",
                        self.successor(predecessor, ordinal, case.target, &location)?
                    ));
                }
                let default =
                    self.successor(predecessor, cases.len(), *default_target, &location)?;
                emit_switch(output, &selector, selector_ty, &default, &rendered);
            }
            Terminator::IntegerSwitch {
                selector,
                cases,
                default_target,
                ..
            } => {
                let (selector, _, selector_ty) = self.typed_operand(*selector, &location)?;
                if integer_width(selector_ty).is_none() {
                    return Err(unsupported(location, "switch selector must be an integer"));
                }
                let mut rendered = Vec::with_capacity(cases.len());
                for (ordinal, case) in cases.iter().enumerate() {
                    let value = integer_switch_case_value(&case.value)
                        .filter(|_| mlir_type(&case.value.ty()) == Some(selector_ty))
                        .ok_or_else(|| {
                            unsupported(
                                location.clone(),
                                format!(
                                    "integer switch case {:?} does not match the selector type",
                                    case.value
                                ),
                            )
                        })?;
                    rendered.push(format!(
                        "{value}: {}",
                        self.successor(predecessor, ordinal, case.target, &location)?
                    ));
                }
                let default =
                    self.successor(predecessor, cases.len(), *default_target, &location)?;
                emit_switch(output, &selector, selector_ty, &default, &rendered);
            }
            Terminator::Return { values } => {
                let signature = &self.lowerer.function.signature.results;
                if values.len() != signature.len() {
                    return Err(unsupported(
                        location,
                        "return arity differs from the function signature",
                    ));
                }
                let mut returned = Vec::with_capacity(values.len());
                for (value, expected) in values.iter().zip(signature) {
                    let (name, _, ty) = self.typed_operand(*value, &location)?;
                    if mlir_type_at(expected, &location)? != ty {
                        return Err(unsupported(
                            location,
                            "returned value type differs from the function signature",
                        ));
                    }
                    returned.push((name, ty));
                }
                match returned.as_slice() {
                    [] => writeln!(output, "    llvm.return").unwrap(),
                    [(name, ty)] => writeln!(output, "    llvm.return {name} : {ty}").unwrap(),
                    _ => {
                        let types = returned.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
                        let aggregate = struct_type(&types);
                        let mut previous = format!("%helper.return.{}.poison", predecessor.0);
                        writeln!(output, "    {previous} = llvm.mlir.poison : {aggregate}")
                            .unwrap();
                        for (index, (name, _)) in returned.iter().enumerate() {
                            let next = format!("%helper.return.{}.{index}", predecessor.0);
                            writeln!(
                                output,
                                "    {next} = llvm.insertvalue {name}, {previous}[{index}] : {aggregate}"
                            )
                            .unwrap();
                            previous = next;
                        }
                        writeln!(output, "    llvm.return {previous} : {aggregate}").unwrap();
                    }
                }
            }
            Terminator::Unreachable => writeln!(output, "    llvm.unreachable").unwrap(),
        }
        Ok(())
    }

    fn emit_operation(
        &mut self,
        output: &mut dyn fmt::Write,
        block: BlockId,
        operation_index: usize,
        operation: &Operation,
    ) -> Result<(), LoweringErrors> {
        let location = self.operation_location(block, operation_index);
        let single_result = || match operation.results.as_slice() {
            [result] => Ok(result),
            _ => Err(unsupported(
                location.clone(),
                "operation must define exactly one result",
            )),
        };
        match &operation.kind {
            OperationKind::Constant(_) => {}
            OperationKind::SliceLength { slice } => {
                let result = single_result()?;
                let (_, length) = self.slice_operand(*slice, &location)?;
                let zero = self.immediate(output, 0, INDEX_TY, &location)?;
                writeln!(
                    output,
                    "    {} = llvm.add {length}, {zero} : {INDEX_TY}",
                    value_name(result.id)
                )
                .unwrap();
            }
            OperationKind::SliceData { slice } => {
                let result = single_result()?;
                let (data, _) = self.slice_operand(*slice, &location)?;
                let zero = self.immediate(output, 0, INDEX_TY, &location)?;
                writeln!(
                    output,
                    "    {} = llvm.getelementptr {data}[{zero}] : ({GLOBAL_PTR}, {INDEX_TY}) -> {GLOBAL_PTR}, i8",
                    value_name(result.id)
                )
                .unwrap();
            }
            OperationKind::Intrinsic(intrinsic) => {
                let result = single_result()?;
                if mlir_type_at(&result.ty, &location)? != INDEX_TY {
                    return Err(unsupported(location, "intrinsic results must be i64"));
                }
                self.emit_intrinsic(output, &value_name(result.id), intrinsic.kind, &location)?;
            }
            OperationKind::Unary { op, operand } => {
                let result = value_name(single_result()?.id);
                let (operand, ty, mlir) = self.typed_operand(*operand, &location)?;
                let style = unary_lowering_style(*op, ty).ok_or_else(|| {
                    unsupported(location.clone(), format!("unary {op:?} on {ty:?}"))
                })?;
                match style {
                    LlvmUnaryStyle::FloatNegate => {
                        writeln!(output, "    {result} = llvm.fneg {operand} : {mlir}").unwrap();
                    }
                    LlvmUnaryStyle::SignedNegate => {
                        let zero = self.immediate(output, 0, mlir, &location)?;
                        writeln!(output, "    {result} = llvm.sub {zero}, {operand} : {mlir}")
                            .unwrap();
                    }
                    LlvmUnaryStyle::BooleanNot => {
                        if mlir != "i1" {
                            return Err(unsupported(location, "boolean not requires i1"));
                        }
                        let one = self.immediate_true(output);
                        writeln!(output, "    {result} = llvm.xor {operand}, {one} : i1").unwrap();
                    }
                    LlvmUnaryStyle::IntegerNot => {
                        let ones = self.immediate(output, -1, mlir, &location)?;
                        writeln!(output, "    {result} = llvm.xor {operand}, {ones} : {mlir}")
                            .unwrap();
                    }
                }
            }
            OperationKind::Binary {
                op: BinaryOp::Checked(operator),
                lhs,
                rhs,
            } => {
                let [value, overflow] = operation.results.as_slice() else {
                    return Err(unsupported(
                        location,
                        "checked arithmetic must define value and overflow results",
                    ));
                };
                let (lhs, lhs_ty, mlir) = self.typed_operand(*lhs, &location)?;
                let (rhs, _, rhs_mlir) = self.typed_operand(*rhs, &location)?;
                let scalar = lhs_ty
                    .as_scalar()
                    .filter(|scalar| scalar.is_integer())
                    .ok_or_else(|| {
                        unsupported(location.clone(), "checked arithmetic requires integers")
                    })?;
                if rhs_mlir != mlir || integer_width(mlir).is_none_or(|width| width < 8) {
                    return Err(unsupported(
                        location,
                        "checked arithmetic operands must share one integer type",
                    ));
                }
                let signed = scalar.is_signed_integer();
                let name = match (operator, signed) {
                    (CheckedBinaryOperator::Add, true) => "sadd",
                    (CheckedBinaryOperator::Add, false) => "uadd",
                    (CheckedBinaryOperator::Subtract, true) => "ssub",
                    (CheckedBinaryOperator::Subtract, false) => "usub",
                    (CheckedBinaryOperator::Multiply, true) => "smul",
                    (CheckedBinaryOperator::Multiply, false) => "umul",
                };
                let pair = format!("%checked.{}.{}", block.0, operation_index);
                let aggregate = struct_type(&[mlir, "i1"]);
                writeln!(
                    output,
                    "    {pair} = \"llvm.intr.{name}.with.overflow\"({lhs}, {rhs}) : ({mlir}, {mlir}) -> {aggregate}"
                )
                .unwrap();
                writeln!(
                    output,
                    "    {} = llvm.extractvalue {pair}[0] : {aggregate}",
                    value_name(value.id)
                )
                .unwrap();
                writeln!(
                    output,
                    "    {} = llvm.extractvalue {pair}[1] : {aggregate}",
                    value_name(overflow.id)
                )
                .unwrap();
            }
            OperationKind::Binary { op, lhs, rhs } => {
                let result = value_name(single_result()?.id);
                let (lhs, lhs_ty, mlir) = self.typed_operand(*lhs, &location)?;
                let (rhs, _, rhs_mlir) = self.typed_operand(*rhs, &location)?;
                if rhs_mlir != mlir {
                    return Err(unsupported(
                        location,
                        "binary operands must share one MLIR type",
                    ));
                }
                let opcode = mlir_binary_opcode(*op, lhs_ty).ok_or_else(|| {
                    unsupported(location.clone(), format!("binary {op:?} on {lhs_ty:?}"))
                })?;
                writeln!(output, "    {result} = llvm.{opcode} {lhs}, {rhs} : {mlir}").unwrap();
            }
            OperationKind::Compare {
                predicate,
                lhs,
                rhs,
            } => {
                let result = value_name(single_result()?.id);
                let (lhs, lhs_ty, mlir) = self.typed_operand(*lhs, &location)?;
                let (rhs, _, rhs_mlir) = self.typed_operand(*rhs, &location)?;
                if rhs_mlir != mlir || lhs_ty.as_scalar().is_none() {
                    return Err(unsupported(
                        location,
                        "comparison operands must share one scalar type",
                    ));
                }
                writeln!(
                    output,
                    "    {result} = llvm.{} \"{}\" {lhs}, {rhs} : {mlir}",
                    compare_opcode(lhs_ty),
                    compare_predicate(*predicate, lhs_ty)
                )
                .unwrap();
            }
            OperationKind::Select {
                condition,
                true_value,
                false_value,
            } => {
                let result = value_name(single_result()?.id);
                let (condition, _, condition_ty) = self.typed_operand(*condition, &location)?;
                let (true_value, _, mlir) = self.typed_operand(*true_value, &location)?;
                let (false_value, _, false_mlir) = self.typed_operand(*false_value, &location)?;
                if condition_ty != "i1" || false_mlir != mlir {
                    return Err(unsupported(
                        location,
                        "select operand types are inconsistent",
                    ));
                }
                writeln!(
                    output,
                    "    {result} = llvm.select {condition}, {true_value}, {false_value} : i1, {mlir}"
                )
                .unwrap();
            }
            OperationKind::Cast { kind, value, to } => {
                let result = value_name(single_result()?.id);
                let (value, from, from_mlir) = self.typed_operand(*value, &location)?;
                let to_mlir = mlir_type_at(to, &location)?;
                match kind {
                    CastKind::FloatToInteger => {
                        let scalar = to
                            .as_scalar()
                            .filter(|scalar| scalar.is_integer())
                            .ok_or_else(|| {
                                unsupported(
                                    location.clone(),
                                    "float-to-integer target must be an integer",
                                )
                            })?;
                        if from_mlir != "f32" {
                            return Err(unsupported(
                                location,
                                "float-to-integer casts are admitted only from f32",
                            ));
                        }
                        writeln!(
                            output,
                            "    {result} = llvm.call_intrinsic \"{}\"({value}) : (f32) -> {to_mlir}",
                            saturating_float_to_integer_intrinsic_name(scalar)
                        )
                        .unwrap();
                    }
                    CastKind::RestrictPointerAccess => {
                        if !matches!(from, Type::Pointer(_)) || to_mlir != from_mlir {
                            return Err(unsupported(
                                location,
                                "pointer access restriction must preserve the pointer type",
                            ));
                        }
                        let one = self.immediate_true(output);
                        writeln!(
                            output,
                            "    {result} = llvm.select {one}, {value}, {value} : i1, {from_mlir}"
                        )
                        .unwrap();
                    }
                    CastKind::Bitcast if from_mlir == to_mlir => {
                        if integer_width(from_mlir).is_none() || from_mlir == "i1" {
                            return Err(unsupported(
                                location,
                                "same-type bitcasts are admitted only for integers",
                            ));
                        }
                        let zero = self.immediate(output, 0, from_mlir, &location)?;
                        writeln!(
                            output,
                            "    {result} = llvm.add {value}, {zero} : {from_mlir}"
                        )
                        .unwrap();
                    }
                    CastKind::PointerToGeneric
                    | CastKind::SliceToGeneric
                    | CastKind::FloatExtend
                    | CastKind::FloatTruncate => {
                        return Err(unsupported(location, format!("{kind:?} casts")));
                    }
                    CastKind::IntegerToFloat if !matches!(to_mlir, "f32" | "f64") => {
                        return Err(unsupported(
                            location,
                            "integer-to-float casts must produce f32 or f64",
                        ));
                    }
                    CastKind::Truncate
                    | CastKind::ZeroExtend
                    | CastKind::SignExtend
                    | CastKind::IntegerToFloat
                    | CastKind::Bitcast => {
                        writeln!(
                            output,
                            "    {result} = llvm.{} {value} : {from_mlir} to {to_mlir}",
                            cast_opcode(*kind, from)
                        )
                        .unwrap();
                    }
                }
            }
            OperationKind::GetElementPointer { base, offset } => {
                let result = value_name(single_result()?.id);
                let (base, base_ty, base_mlir) = self.typed_operand(*base, &location)?;
                let (offset, _, offset_mlir) = self.typed_operand(*offset, &location)?;
                let Type::Pointer(pointer) = base_ty else {
                    return Err(unsupported(location, "GEP base must be a pointer"));
                };
                let pointee = mlir_type_at(&pointer.pointee, &location)?;
                if integer_width(offset_mlir).is_none() {
                    return Err(unsupported(location, "GEP offset must be an integer"));
                }
                writeln!(
                    output,
                    "    {result} = llvm.getelementptr {base}[{offset}] : ({base_mlir}, {offset_mlir}) -> {base_mlir}, {pointee}"
                )
                .unwrap();
            }
            OperationKind::Load { pointer, access } => {
                let result = value_name(single_result()?.id);
                let (pointer, pointee, pointer_mlir) = self.pointer_operand(*pointer, &location)?;
                check_alignment(access.alignment, &location)?;
                let volatile = if access.volatile { "volatile " } else { "" };
                writeln!(
                    output,
                    "    {result} = llvm.load {volatile}{pointer} <alignment = {}> : {pointer_mlir} -> {pointee}",
                    access.alignment
                )
                .unwrap();
            }
            OperationKind::Store {
                pointer,
                value,
                access,
            } => {
                let (pointer, pointee, pointer_mlir) = self.pointer_operand(*pointer, &location)?;
                let (value, _, value_mlir) = self.typed_operand(*value, &location)?;
                check_alignment(access.alignment, &location)?;
                if value_mlir != pointee {
                    return Err(unsupported(
                        location,
                        "stored value type differs from the pointee",
                    ));
                }
                let volatile = if access.volatile { "volatile " } else { "" };
                writeln!(
                    output,
                    "    llvm.store {volatile}{value}, {pointer} <alignment = {}> : {pointee}, {pointer_mlir}",
                    access.alignment
                )
                .unwrap();
            }
            OperationKind::GuardedLoad {
                pointer,
                predicate,
                fallback,
                access,
            } => {
                let result = value_name(single_result()?.id);
                let (pointer, pointee, pointer_mlir) = self.pointer_operand(*pointer, &location)?;
                let (predicate, _, predicate_ty) = self.typed_operand(*predicate, &location)?;
                let (fallback, _, fallback_ty) = self.typed_operand(*fallback, &location)?;
                check_alignment(access.alignment, &location)?;
                if predicate_ty != "i1" || fallback_ty != pointee {
                    return Err(unsupported(
                        location,
                        "guarded load predicate or fallback type is inconsistent",
                    ));
                }
                let volatile = if access.volatile { "volatile " } else { "" };
                let true_label = guarded_load_true_label(block, operation_index);
                let false_label = guarded_load_false_label(block, operation_index);
                let merge_label = guarded_load_merge_label(block, operation_index);
                writeln!(
                    output,
                    "    llvm.cond_br {predicate}, ^{true_label}, ^{false_label}"
                )
                .unwrap();
                writeln!(output, "  ^{true_label}:").unwrap();
                writeln!(
                    output,
                    "    {result}.loaded = llvm.load {volatile}{pointer} <alignment = {}> : {pointer_mlir} -> {pointee}",
                    access.alignment
                )
                .unwrap();
                writeln!(
                    output,
                    "    llvm.br ^{merge_label}({result}.loaded : {pointee})"
                )
                .unwrap();
                writeln!(output, "  ^{false_label}:").unwrap();
                writeln!(output, "    llvm.br ^{merge_label}({fallback} : {pointee})").unwrap();
                writeln!(output, "  ^{merge_label}({result}: {pointee}):").unwrap();
            }
            OperationKind::GuardedStore {
                pointer,
                predicate,
                value,
                access,
            } => {
                let (pointer, pointee, pointer_mlir) = self.pointer_operand(*pointer, &location)?;
                let (predicate, _, predicate_ty) = self.typed_operand(*predicate, &location)?;
                let (value, _, value_mlir) = self.typed_operand(*value, &location)?;
                check_alignment(access.alignment, &location)?;
                if predicate_ty != "i1" || value_mlir != pointee {
                    return Err(unsupported(
                        location,
                        "guarded store predicate or value type is inconsistent",
                    ));
                }
                let volatile = if access.volatile { "volatile " } else { "" };
                let true_label = guarded_store_true_label(block, operation_index);
                let merge_label = guarded_store_merge_label(block, operation_index);
                writeln!(
                    output,
                    "    llvm.cond_br {predicate}, ^{true_label}, ^{merge_label}"
                )
                .unwrap();
                writeln!(output, "  ^{true_label}:").unwrap();
                writeln!(
                    output,
                    "    llvm.store {volatile}{value}, {pointer} <alignment = {}> : {pointee}, {pointer_mlir}",
                    access.alignment
                )
                .unwrap();
                writeln!(output, "    llvm.br ^{merge_label}").unwrap();
                writeln!(output, "  ^{merge_label}:").unwrap();
            }
            OperationKind::Alloca {
                element,
                count,
                address_space,
                alignment,
            } => {
                let result = value_name(single_result()?.id);
                if *address_space != KernelAddressSpace::Private {
                    return Err(unsupported(location, "allocas must be private"));
                }
                check_alignment(*alignment, &location)?;
                let element = mlir_type_at(element, &location)?;
                let (count, count_ty) = match count {
                    Some(count) => {
                        let (count, _, count_ty) = self.typed_operand(*count, &location)?;
                        if integer_width(count_ty).is_none() {
                            return Err(unsupported(location, "alloca count must be an integer"));
                        }
                        (count, count_ty)
                    }
                    None => (self.immediate(output, 1, "i32", &location)?, "i32"),
                };
                writeln!(
                    output,
                    "    {result} = llvm.alloca {count} x {element} {{alignment = {alignment} : i64}} : ({count_ty}) -> !llvm.ptr<5>"
                )
                .unwrap();
            }
            OperationKind::Call { callee, arguments } => {
                self.emit_call(output, block, operation_index, operation, callee, arguments)?;
            }
            OperationKind::Atomic(atomic) => {
                self.emit_atomic(output, operation, atomic, &location)?;
            }
            OperationKind::Fence(fence) => {
                emit_mlir_fence(
                    output,
                    fence.memory_scope,
                    fence.semantics.ordering,
                    &location,
                )?;
            }
            OperationKind::WorkgroupBarrier(barrier) => {
                if !self.lowerer.target.requires_physical_workgroup_barrier() {
                    return Err(unsupported(
                        location,
                        "only the physical s_barrier policy is representable",
                    ));
                }
                match barrier.semantics.ordering {
                    MemoryOrdering::Acquire => {}
                    MemoryOrdering::Release | MemoryOrdering::AcquireRelease => {
                        emit_mlir_fence(
                            output,
                            barrier.memory_scope,
                            MemoryOrdering::Release,
                            &location,
                        )?;
                    }
                    MemoryOrdering::SequentiallyConsistent => {
                        emit_mlir_fence(
                            output,
                            barrier.memory_scope,
                            MemoryOrdering::SequentiallyConsistent,
                            &location,
                        )?;
                    }
                    MemoryOrdering::Relaxed => {
                        return Err(unsupported(location, "relaxed workgroup barriers"));
                    }
                }
                writeln!(
                    output,
                    "    llvm.inline_asm has_side_effects \"s_barrier\", \"\" : () -> ()"
                )
                .unwrap();
                match barrier.semantics.ordering {
                    MemoryOrdering::Release => {}
                    MemoryOrdering::Acquire | MemoryOrdering::AcquireRelease => {
                        emit_mlir_fence(
                            output,
                            barrier.memory_scope,
                            MemoryOrdering::Acquire,
                            &location,
                        )?;
                    }
                    MemoryOrdering::SequentiallyConsistent => {
                        emit_mlir_fence(
                            output,
                            barrier.memory_scope,
                            MemoryOrdering::SequentiallyConsistent,
                            &location,
                        )?;
                    }
                    MemoryOrdering::Relaxed => {
                        return Err(unsupported(location, "relaxed workgroup barriers"));
                    }
                }
            }
            OperationKind::WorkgroupMemory(memory) => {
                let result = single_result()?;
                let kernel = self
                    .lowerer
                    .kernel
                    .ok_or_else(|| unsupported(location.clone(), "LDS requires a kernel"))?;
                let symbol = lds_symbol(kernel, result.id);
                let symbol = symbol.strip_prefix('@').unwrap_or(&symbol);
                check_symbol(symbol, &location)?;
                let element = mlir_type_at(&memory.element, &location)?;
                let elements = match memory.extent {
                    WorkgroupMemoryExtent::Static(elements) => elements,
                    WorkgroupMemoryExtent::Dynamic | WorkgroupMemoryExtent::DynamicAtLeast(_) => 0,
                };
                let name = value_name(result.id);
                writeln!(
                    output,
                    "    {name}.lds = llvm.mlir.addressof @{symbol} : !llvm.ptr<3>"
                )
                .unwrap();
                writeln!(
                    output,
                    "    {name} = llvm.getelementptr {name}.lds[0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<{elements} x {element}>"
                )
                .unwrap();
            }
            other => {
                return Err(unsupported(
                    location,
                    format!(
                        "operation kind {} is outside the reviewed subset",
                        operation_kind_name(other)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Resolves a pointer operand to `(name, pointee MLIR type, pointer MLIR type)`.
    fn pointer_operand(
        &self,
        value: ValueId,
        location: &LoweringLocation,
    ) -> Result<(String, &'static str, &'static str), LoweringErrors> {
        let (name, ty, mlir) = self.typed_operand(value, location)?;
        let Type::Pointer(pointer) = ty else {
            return Err(unsupported(
                location.clone(),
                "memory access requires a pointer",
            ));
        };
        Ok((name, mlir_type_at(&pointer.pointee, location)?, mlir))
    }

    fn emit_call(
        &mut self,
        output: &mut dyn fmt::Write,
        block: BlockId,
        operation_index: usize,
        operation: &Operation,
        callee: &FunctionId,
        arguments: &[ValueId],
    ) -> Result<(), LoweringErrors> {
        let location = self.operation_location(block, operation_index);
        if AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments).is_some()
            || FloatOperation::from_intrinsic_call(callee, arguments).is_some()
        {
            return Err(unsupported(
                location,
                "diagnostic and float-support intrinsic calls are not supported",
            ));
        }
        let lowerer = self.lowerer;
        let callee_function = lowerer
            .module
            .function(callee)
            .ok_or_else(|| unsupported(location.clone(), "callee is not in the module"))?;
        let symbol = lowerer
            .call_symbols
            .and_then(|symbols| symbols.get(callee))
            .ok_or_else(|| unsupported(location.clone(), "callee has no compiler-module symbol"))?;
        check_symbol(symbol, &location)?;
        let mut names = Vec::new();
        let mut types = Vec::new();
        for argument in arguments {
            self.push_flattened(*argument, &location, &mut names, &mut types)?;
        }
        let convergent = lowerer
            .wave64_helper_context_v1
            .is_some_and(|context| {
                context.call_is_convergent(
                    lowerer.module,
                    lowerer.function,
                    block,
                    operation_index,
                    callee,
                )
            })
            .then_some(" {convergent}")
            .unwrap_or("");
        let results = callee_function
            .signature
            .results
            .iter()
            .map(|ty| mlir_type_at(ty, &location))
            .collect::<Result<Vec<_>, _>>()?;
        if operation.results.len() != results.len() {
            return Err(unsupported(
                location,
                "call result count differs from the callee signature",
            ));
        }
        let call = format!(
            "llvm.call @{symbol}({}){convergent} : ({})",
            names.join(", "),
            types.join(", ")
        );
        match results.as_slice() {
            [] => writeln!(output, "    {call} -> ()").unwrap(),
            [result] => writeln!(
                output,
                "    {} = {call} -> {result}",
                value_name(operation.results[0].id)
            )
            .unwrap(),
            _ => {
                let aggregate = struct_type(&results);
                let pair = format!("%helper.call.{}.{}", block.0, operation_index);
                writeln!(output, "    {pair} = {call} -> {aggregate}").unwrap();
                for (index, result) in operation.results.iter().enumerate() {
                    writeln!(
                        output,
                        "    {} = llvm.extractvalue {pair}[{index}] : {aggregate}",
                        value_name(result.id)
                    )
                    .unwrap();
                }
            }
        }
        Ok(())
    }

    fn emit_atomic(
        &mut self,
        output: &mut dyn fmt::Write,
        operation: &Operation,
        atomic: &Atomic,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let (pointer, pointee, pointer_mlir) = self.pointer_operand(atomic.pointer, location)?;
        check_alignment(atomic.access.alignment, location)?;
        let alignment = atomic.access.alignment;
        let scope = mlir_atomic_sync_scope(atomic.scope).ok_or_else(|| {
            unsupported(location.clone(), format!("atomic scope {:?}", atomic.scope))
        })?;
        let ordering = llvm_atomic_ordering(atomic.ordering);
        let value = |emitter: &Self, value: Option<ValueId>, what: &str| {
            let value = value.ok_or_else(|| {
                unsupported(location.clone(), format!("atomic is missing its {what}"))
            })?;
            let (name, _, ty) = emitter.typed_operand(value, location)?;
            if ty != pointee {
                return Err(unsupported(
                    location.clone(),
                    format!("atomic {what} type differs from the pointee"),
                ));
            }
            Ok(name)
        };
        match atomic.kind {
            AtomicKind::Load => {
                let [result] = operation.results.as_slice() else {
                    return Err(unsupported(
                        location.clone(),
                        "atomic load must define one result",
                    ));
                };
                writeln!(
                    output,
                    "    {} = llvm.load {pointer} atomic{scope} {ordering} <alignment = {alignment}> : {pointer_mlir} -> {pointee}",
                    value_name(result.id)
                )
                .unwrap();
            }
            AtomicKind::Store => {
                let stored = value(self, atomic.value, "stored value")?;
                writeln!(
                    output,
                    "    llvm.store {stored}, {pointer} atomic{scope} {ordering} <alignment = {alignment}> : {pointee}, {pointer_mlir}"
                )
                .unwrap();
            }
            AtomicKind::CompareExchange => {
                let [old, succeeded] = operation.results.as_slice() else {
                    return Err(unsupported(
                        location.clone(),
                        "compare-exchange must define old and success results",
                    ));
                };
                let desired = value(self, atomic.value, "desired value")?;
                let expected = value(self, atomic.compare, "expected value")?;
                let failure = atomic.failure_ordering.ok_or_else(|| {
                    unsupported(
                        location.clone(),
                        "compare-exchange is missing its failure ordering",
                    )
                })?;
                let failure = llvm_atomic_ordering(failure);
                let pair = format!("{}.cmpxchg", value_name(old.id));
                let aggregate = struct_type(&[pointee, "i1"]);
                writeln!(
                    output,
                    "    {pair} = llvm.cmpxchg {pointer}, {expected}, {desired}{scope} {ordering} {failure} <alignment = {alignment}> : {pointer_mlir}, {pointee}"
                )
                .unwrap();
                writeln!(
                    output,
                    "    {} = llvm.extractvalue {pair}[0] : {aggregate}",
                    value_name(old.id)
                )
                .unwrap();
                writeln!(
                    output,
                    "    {} = llvm.extractvalue {pair}[1] : {aggregate}",
                    value_name(succeeded.id)
                )
                .unwrap();
            }
            kind => {
                let [result] = operation.results.as_slice() else {
                    return Err(unsupported(
                        location.clone(),
                        "atomic RMW must define one result",
                    ));
                };
                let Type::Pointer(pointer_ty) = self.operand(atomic.pointer, location)?.1 else {
                    return Err(unsupported(location.clone(), "atomic requires a pointer"));
                };
                let opcode = pointer_ty
                    .pointee
                    .as_scalar()
                    .and_then(|scalar| mlir_atomic_rmw_opcode(kind, scalar))
                    .ok_or_else(|| {
                        unsupported(
                            location.clone(),
                            format!("atomic RMW {kind:?} on {:?}", pointer_ty.pointee),
                        )
                    })?;
                let operand = value(self, atomic.value, "RMW operand")?;
                writeln!(
                    output,
                    "    {} = llvm.atomicrmw {opcode} {pointer}, {operand}{scope} {ordering} <alignment = {alignment}> : {pointer_mlir}, {pointee}",
                    value_name(result.id)
                )
                .unwrap();
            }
        }
        Ok(())
    }

    fn dispatch_field(
        &mut self,
        output: &mut dyn fmt::Write,
        dispatch: &str,
        prefix: &str,
        offset: u32,
        loaded: &str,
        alignment: u32,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let offset = self.immediate(output, offset, INDEX_TY, location)?;
        writeln!(
            output,
            "    {prefix}.ptr = llvm.getelementptr inbounds {dispatch}[{offset}] : ({CONSTANT_PTR}, {INDEX_TY}) -> {CONSTANT_PTR}, i8"
        )
        .unwrap();
        writeln!(
            output,
            "    {prefix}.{loaded} = llvm.load {prefix}.ptr <alignment = {alignment}> : {CONSTANT_PTR} -> {loaded}"
        )
        .unwrap();
        Ok(())
    }

    fn emit_dispatch_ptr(output: &mut dyn fmt::Write, result: &str) {
        writeln!(
            output,
            "    {result}.dispatch = llvm.call_intrinsic \"{}\"() : () -> {CONSTANT_PTR}",
            AmdgcnIntrinsic::DispatchPtr.llvm_name()
        )
        .unwrap();
    }

    fn emit_intrinsic(
        &mut self,
        output: &mut dyn fmt::Write,
        result: &str,
        kind: IntrinsicKind,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let axis_suffix = |axis: Axis| match axis {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        };
        match kind {
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            } => self.emit_logical_global_id(output, result, location)?,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            } => {
                writeln!(output, "    {result}.local.i32 = rocdl.workitem.id.x : i32").unwrap();
                writeln!(
                    output,
                    "    {result}.local = llvm.zext {result}.local.i32 : i32 to i64"
                )
                .unwrap();
                let zero = self.immediate(output, 0, INDEX_TY, location)?;
                writeln!(
                    output,
                    "    {result} = llvm.add {result}.local, {zero} : i64"
                )
                .unwrap();
            }
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Workgroup,
                axis,
            } => {
                writeln!(
                    output,
                    "    {result}.group.i32 = rocdl.workgroup.id.{} : i32",
                    axis_suffix(axis)
                )
                .unwrap();
                writeln!(
                    output,
                    "    {result}.group = llvm.zext {result}.group.i32 : i32 to i64"
                )
                .unwrap();
                let zero = self.immediate(output, 0, INDEX_TY, location)?;
                writeln!(
                    output,
                    "    {result} = llvm.add {result}.group, {zero} : i64"
                )
                .unwrap();
            }
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::WorkgroupSize,
                axis,
            } => {
                if let Some(size) = self.lowerer.workgroup_size {
                    let extent = match axis {
                        Axis::X => size.x,
                        Axis::Y => size.y,
                        Axis::Z => size.z,
                    };
                    let extent = self.immediate(output, extent, INDEX_TY, location)?;
                    let zero = self.immediate(output, 0, INDEX_TY, location)?;
                    writeln!(output, "    {result} = llvm.add {extent}, {zero} : i64").unwrap();
                } else {
                    let offset = match axis {
                        Axis::X => 4,
                        Axis::Y => 6,
                        Axis::Z => 8,
                    };
                    Self::emit_dispatch_ptr(output, result);
                    let dispatch = format!("{result}.dispatch");
                    let prefix = format!("{result}.workgroup");
                    self.dispatch_field(output, &dispatch, &prefix, offset, "i16", 2, location)?;
                    writeln!(output, "    {result} = llvm.zext {prefix}.i16 : i16 to i64").unwrap();
                }
            }
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::WorkgroupCount,
                axis,
            } => {
                let (workgroup_offset, grid_offset, static_extent) = match axis {
                    Axis::X => (4, 12, self.lowerer.workgroup_size.map(|size| size.x)),
                    Axis::Y => (6, 16, self.lowerer.workgroup_size.map(|size| size.y)),
                    Axis::Z => (8, 20, self.lowerer.workgroup_size.map(|size| size.z)),
                };
                Self::emit_dispatch_ptr(output, result);
                let dispatch = format!("{result}.dispatch");
                let grid = format!("{result}.grid");
                self.dispatch_field(output, &dispatch, &grid, grid_offset, "i32", 4, location)?;
                writeln!(output, "    {grid} = llvm.zext {grid}.i32 : i32 to i64").unwrap();
                if let Some(extent) = static_extent {
                    let bias = extent.checked_sub(1).ok_or_else(|| {
                        unsupported(location.clone(), "workgroup extent must be nonzero")
                    })?;
                    let bias = self.immediate(output, bias, INDEX_TY, location)?;
                    writeln!(
                        output,
                        "    {result}.rounded = llvm.add {grid}, {bias} : i64"
                    )
                    .unwrap();
                    let extent = self.immediate(output, extent, INDEX_TY, location)?;
                    writeln!(
                        output,
                        "    {result} = llvm.udiv {result}.rounded, {extent} : i64"
                    )
                    .unwrap();
                } else {
                    let workgroup = format!("{result}.workgroup");
                    self.dispatch_field(
                        output,
                        &dispatch,
                        &workgroup,
                        workgroup_offset,
                        "i16",
                        2,
                        location,
                    )?;
                    writeln!(
                        output,
                        "    {workgroup} = llvm.zext {workgroup}.i16 : i16 to i64"
                    )
                    .unwrap();
                    let one = self.immediate(output, 1, INDEX_TY, location)?;
                    writeln!(
                        output,
                        "    {workgroup}.minus_one = llvm.sub {workgroup}, {one} : i64"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "    {result}.rounded = llvm.add {grid}, {workgroup}.minus_one : i64"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "    {result} = llvm.udiv {result}.rounded, {workgroup} : i64"
                    )
                    .unwrap();
                }
            }
            other => {
                return Err(unsupported(
                    location.clone(),
                    format!("intrinsic {other:?} is outside the reviewed subset"),
                ));
            }
        }
        Ok(())
    }

    fn emit_logical_global_id(
        &mut self,
        output: &mut dyn fmt::Write,
        result: &str,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let kernel = self.lowerer.kernel.ok_or_else(|| {
            unsupported(
                location.clone(),
                "global invocation index requires a kernel entry",
            )
        })?;
        let workgroup = self
            .lowerer
            .required_workgroup_size("global invocation index requires a workgroup size")?;
        let rank = usize::from(kernel.domain.rank());
        if rank == 1 {
            writeln!(output, "    {result}.local.i32 = rocdl.workitem.id.x : i32").unwrap();
            writeln!(
                output,
                "    {result}.group.i32 = rocdl.workgroup.id.x : i32"
            )
            .unwrap();
            writeln!(
                output,
                "    {result}.local = llvm.zext {result}.local.i32 : i32 to i64"
            )
            .unwrap();
            writeln!(
                output,
                "    {result}.group = llvm.zext {result}.group.i32 : i32 to i64"
            )
            .unwrap();
            let extent = self.immediate(output, workgroup.x, INDEX_TY, location)?;
            writeln!(
                output,
                "    {result}.base = llvm.mul {result}.group, {extent} : i64"
            )
            .unwrap();
            writeln!(
                output,
                "    {result} = llvm.add {result}.base, {result}.local : i64"
            )
            .unwrap();
            return Ok(());
        }
        if !(2..=3).contains(&rank) {
            return Err(unsupported(
                location.clone(),
                format!("launch domain rank {rank} is not representable"),
            ));
        }
        for (suffix, extent) in [("x", workgroup.x), ("y", workgroup.y), ("z", workgroup.z)]
            .into_iter()
            .take(rank)
        {
            let dim = format!("{result}.{suffix}");
            writeln!(
                output,
                "    {dim}.local.i32 = rocdl.workitem.id.{suffix} : i32"
            )
            .unwrap();
            writeln!(
                output,
                "    {dim}.group.i32 = rocdl.workgroup.id.{suffix} : i32"
            )
            .unwrap();
            writeln!(
                output,
                "    {dim}.local = llvm.zext {dim}.local.i32 : i32 to i64"
            )
            .unwrap();
            writeln!(
                output,
                "    {dim}.group = llvm.zext {dim}.group.i32 : i32 to i64"
            )
            .unwrap();
            let extent = self.immediate(output, extent, INDEX_TY, location)?;
            writeln!(
                output,
                "    {dim}.base = llvm.mul {dim}.group, {extent} : i64"
            )
            .unwrap();
            writeln!(output, "    {dim} = llvm.add {dim}.base, {dim}.local : i64").unwrap();
        }
        Self::emit_dispatch_ptr(output, result);
        let dispatch = format!("{result}.dispatch");
        let grid_x = format!("{result}.grid.x");
        self.dispatch_field(output, &dispatch, &grid_x, 12, "i32", 4, location)?;
        writeln!(output, "    {grid_x} = llvm.zext {grid_x}.i32 : i32 to i64").unwrap();
        if rank == 2 {
            writeln!(
                output,
                "    {result}.row = llvm.mul {result}.y, {grid_x} : i64"
            )
            .unwrap();
            writeln!(
                output,
                "    {result} = llvm.add {result}.row, {result}.x : i64"
            )
            .unwrap();
            return Ok(());
        }
        let grid_y = format!("{result}.grid.y");
        self.dispatch_field(output, &dispatch, &grid_y, 16, "i32", 4, location)?;
        writeln!(output, "    {grid_y} = llvm.zext {grid_y}.i32 : i32 to i64").unwrap();
        writeln!(
            output,
            "    {result}.plane = llvm.mul {result}.z, {grid_y} : i64"
        )
        .unwrap();
        writeln!(
            output,
            "    {result}.plane_row = llvm.add {result}.plane, {result}.y : i64"
        )
        .unwrap();
        writeln!(
            output,
            "    {result}.plane_row_scaled = llvm.mul {result}.plane_row, {grid_x} : i64"
        )
        .unwrap();
        writeln!(
            output,
            "    {result} = llvm.add {result}.plane_row_scaled, {result}.x : i64"
        )
        .unwrap();
        Ok(())
    }
}

fn emit_switch(
    output: &mut dyn fmt::Write,
    selector: &str,
    selector_ty: &str,
    default: &str,
    cases: &[String],
) {
    if cases.is_empty() {
        writeln!(
            output,
            "    llvm.switch {selector} : {selector_ty}, {default} []"
        )
        .unwrap();
        return;
    }
    writeln!(
        output,
        "    llvm.switch {selector} : {selector_ty}, {default} ["
    )
    .unwrap();
    for (index, case) in cases.iter().enumerate() {
        let separator = if index + 1 == cases.len() { "" } else { "," };
        writeln!(output, "      {case}{separator}").unwrap();
    }
    writeln!(output, "    ]").unwrap();
}

fn emit_mlir_fence(
    output: &mut dyn fmt::Write,
    scope: SynchronizationScope,
    ordering: MemoryOrdering,
    location: &LoweringLocation,
) -> Result<(), LoweringErrors> {
    let ordering = match ordering {
        MemoryOrdering::Acquire => "acquire",
        MemoryOrdering::Release => "release",
        MemoryOrdering::AcquireRelease => "acq_rel",
        MemoryOrdering::SequentiallyConsistent => "seq_cst",
        MemoryOrdering::Relaxed => {
            return Err(unsupported(location.clone(), "relaxed fences"));
        }
    };
    let scope = mlir_fence_sync_scope(scope)
        .ok_or_else(|| unsupported(location.clone(), format!("fence scope {scope:?}")))?;
    writeln!(output, "    llvm.fence{scope} {ordering}").unwrap();
    Ok(())
}

fn operation_kind_name(kind: &OperationKind) -> String {
    let debug = format!("{kind:?}");
    debug
        .split(|character: char| !character.is_ascii_alphanumeric())
        .next()
        .unwrap_or("unknown")
        .to_owned()
}
