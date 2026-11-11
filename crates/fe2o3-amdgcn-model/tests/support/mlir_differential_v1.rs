//! Shared harness for the engineering-only MLIR `llvm`/`rocdl` compiler-module emitter.
//!
//! Included with `#[path]` by several integration-test crates so each can reuse its own module
//! builders. LLVM-free structural checks always run, including the ordered argument names the
//! worker restores from `loc("...")`. The differential check runs only when `FE2O3_LLVM_BIN` names
//! a directory that holds `mlir-translate`, `opt`, and `llc`; without it each test binary prints
//! one notice to stderr, and `FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1` turns the skip into a failure.
//! The differential translates the MLIR, upgrades both modules through bitcode as the worker does,
//! and requires equal canonical IR ([`canonical_ir`]); it then runs both through `opt -O3` and
//! `llc -O3` and requires equal normalized assembly ([`normalize_assembly`]), kernel descriptor
//! and HSA metadata included. When `FE2O3_MLIR_DUMP_DIR` names a directory, every admitted pair is
//! also written there as `{file}-{name}.{cpu}.ll` and `.mlir` for the worker's corpus mode
//! (`fe2o3-worker-mlir-input-tests --corpus DIR`), which checks every pair's closure and argument
//! names after the worker's own translation and links each pair without imports through the full
//! V2 pipeline to compare the HSACO bytes.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};

use fe2o3_amdgcn_model::{
    GFX942_XNACK_MINUS_DATA_LAYOUT, LoweringDiagnosticCode, LoweringErrors,
    lower_compiler_module_to_gfx942_xnack_minus_llvm_ir,
    lower_compiler_module_to_gfx942_xnack_minus_mlir_v1,
    lower_compiler_module_to_gfx950_xnack_minus_llvm_ir,
    lower_compiler_module_to_gfx950_xnack_minus_mlir_v1,
};
use fe2o3_kernel_ir::{
    Module, TargetCapability, WaveWidth, gfx942_xnack_minus_target_capability,
    gfx950_xnack_minus_target_capability,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlirTarget {
    Gfx942,
    Gfx950,
}

pub const BOTH_TARGETS: [MlirTarget; 2] = [MlirTarget::Gfx942, MlirTarget::Gfx950];

impl MlirTarget {
    pub fn cpu(self) -> &'static str {
        match self {
            Self::Gfx942 => "gfx942",
            Self::Gfx950 => "gfx950",
        }
    }
}

/// Binds every module, kernel, and defined function to the exact xnack- target capability.
pub fn exact(mut module: Module, target: MlirTarget) -> Module {
    let capability = match target {
        MlirTarget::Gfx942 => gfx942_xnack_minus_target_capability(),
        MlirTarget::Gfx950 => gfx950_xnack_minus_target_capability(),
    };
    module.required_capabilities.retain(|existing| {
        *existing != gfx942_xnack_minus_target_capability()
            && *existing != gfx950_xnack_minus_target_capability()
    });
    module.required_capabilities.insert(capability.clone());
    for function in &mut module.functions {
        function.required_capabilities.retain(|existing| {
            *existing != gfx942_xnack_minus_target_capability()
                && *existing != gfx950_xnack_minus_target_capability()
        });
        if function.body.is_some() {
            function.required_capabilities.insert(capability.clone());
        }
    }
    for kernel in &mut module.kernels {
        kernel.required_capabilities.retain(|existing| {
            *existing != gfx942_xnack_minus_target_capability()
                && *existing != gfx950_xnack_minus_target_capability()
        });
        kernel.required_capabilities.insert(capability.clone());
    }
    module
}

pub fn text(module: &Module, target: MlirTarget) -> Result<String, LoweringErrors> {
    match target {
        MlirTarget::Gfx942 => lower_compiler_module_to_gfx942_xnack_minus_llvm_ir(module),
        MlirTarget::Gfx950 => lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(module),
    }
}

pub fn mlir(module: &Module, target: MlirTarget) -> Result<String, LoweringErrors> {
    match target {
        MlirTarget::Gfx942 => lower_compiler_module_to_gfx942_xnack_minus_mlir_v1(module),
        MlirTarget::Gfx950 => lower_compiler_module_to_gfx950_xnack_minus_mlir_v1(module),
    }
}

/// LLVM-free invariants every emitted MLIR module must satisfy; `text` is the textual emitter's
/// module for the same input.
pub fn assert_structure(name: &str, text: &str, mlir: &str, target: MlirTarget) {
    let header = format!(
        "module attributes {{llvm.target_triple = \"amdgcn-amd-amdhsa\", llvm.data_layout = \"{GFX942_XNACK_MINUS_DATA_LAYOUT}\"}} {{\n"
    );
    assert!(
        mlir.starts_with(&header),
        "{name}: unexpected module header"
    );
    assert!(
        mlir.ends_with("\n}\n"),
        "{name}: module must close exactly once"
    );
    for forbidden in [
        "llvm.named_metadata",
        "llvm.module_flags",
        "llvm.intr.fptosi.sat",
        "llvm.intr.fptoui.sat",
        "{alignment = ",
    ] {
        let allowed_alloca = forbidden == "{alignment = "
            && mlir
                .lines()
                .filter(|line| line.contains(forbidden))
                .all(|line| line.contains("llvm.alloca"));
        assert!(
            allowed_alloca || !mlir.contains(forbidden),
            "{name}: emitted forbidden construct {forbidden:?}"
        );
    }
    let mut depth = 0_i64;
    for character in mlir.chars() {
        match character {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        assert!(depth >= 0, "{name}: unbalanced braces");
    }
    assert_eq!(depth, 0, "{name}: unbalanced braces");

    let cpu = format!("target_cpu = \"{}\"", target.cpu());
    let mut definitions = BTreeSet::new();
    let mut in_function = false;
    for line in mlir.lines() {
        if line.starts_with("  llvm.func ") {
            let has_body = line.ends_with('{');
            if has_body {
                assert!(
                    line.contains(&cpu),
                    "{name}: definition lacks {cpu}: {line}"
                );
                assert!(
                    line.contains("no_unwind"),
                    "{name}: definition lacks no_unwind"
                );
                assert!(
                    line.contains("denormal_fpenv = #llvm.denormal_fpenv<"),
                    "{name}: definition lacks the IEEE denormal environment"
                );
                if line.contains("amdgpu_kernelcc") {
                    assert!(
                        line.contains("reqd_work_group_size = array<i32: ")
                            && line.contains("[\"amdgpu-flat-work-group-size\", "),
                        "{name}: kernel lacks its workgroup contract: {line}"
                    );
                }
                definitions.clear();
                in_function = true;
            }
            continue;
        }
        if line == "  }" {
            in_function = false;
            continue;
        }
        if !in_function {
            continue;
        }
        let trimmed = line.trim_start();
        let mut define = |value: &str| {
            assert!(
                definitions.insert(value.to_owned()),
                "{name}: SSA value {value} defined twice"
            );
        };
        if let Some(label) = trimmed.strip_prefix('^') {
            assert!(line.starts_with("  ^"), "{name}: misindented block label");
            if let Some((_, arguments)) = label.split_once('(') {
                for argument in arguments.trim_end_matches("):").split(", ") {
                    define(argument.split(':').next().unwrap());
                }
            }
            continue;
        }
        if let Some((value, _)) = trimmed.split_once(" = ")
            && value.starts_with('%')
            && !value.contains(' ')
        {
            define(value);
        }
    }
    assert_parameter_names(name, text, mlir);
}

/// The worker names each LLVM argument after its `loc("...")`, so every MLIR definition must
/// carry, in order, exactly the textual emitter's parameter names, each on the SSA value of the
/// same name.
fn assert_parameter_names(name: &str, text: &str, mlir: &str) {
    let expected = text
        .lines()
        .filter(|line| line.starts_with("define "))
        .map(|line| {
            let (function, parameters) = signature(line);
            let names = split_top_level(parameters, ',')
                .into_iter()
                .map(|parameter| {
                    let mut last = None;
                    rewrite_identifiers(parameter, |token| {
                        if token.starts_with('%') {
                            last = Some(token[1..].to_owned());
                        }
                        None
                    });
                    last.unwrap_or_else(|| panic!("{name}: unnamed parameter in {line}"))
                })
                .collect::<Vec<_>>();
            (function.to_owned(), names)
        })
        .collect::<BTreeMap<_, _>>();
    let actual = mlir
        .lines()
        .filter(|line| line.starts_with("  llvm.func ") && line.ends_with('{'))
        .map(|line| {
            let (function, parameters) = signature(line);
            let names = split_top_level(parameters, ',')
                .into_iter()
                .map(|parameter| {
                    let (value, _) = parameter
                        .split_once(':')
                        .unwrap_or_else(|| panic!("{name}: untyped parameter in {line}"));
                    let label = parameter
                        .rsplit_once("loc(\"")
                        .and_then(|(_, rest)| rest.strip_suffix("\")"))
                        .unwrap_or_else(|| {
                            panic!("{name}: parameter {value} of @{function} has no name location")
                        });
                    assert_eq!(
                        value.trim(),
                        format!("%{label}"),
                        "{name}: @{function} names its location {label:?} on another value"
                    );
                    label.to_owned()
                })
                .collect::<Vec<_>>();
            (function.to_owned(), names)
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        actual, expected,
        "{name}: MLIR definitions and parameter names differ from the textual module"
    );
}

/// The function name and the parameter list of a `define` or `llvm.func` line.
fn signature(line: &str) -> (&str, &str) {
    let at = line.find('@').unwrap();
    let open = at + line[at..].find('(').unwrap();
    let close = matching(line, open);
    (&line[at + 1..open], &line[open + 1..close])
}

/// Index of the bracket that closes the one at `open`, skipping string literals.
fn matching(line: &str, open: usize) -> usize {
    let mut depth = 0_usize;
    let mut quoted = false;
    for (index, character) in line.char_indices().skip_while(|(index, _)| *index < open) {
        match character {
            '"' => quoted = !quoted,
            '(' | '[' | '{' if !quoted => depth += 1,
            ')' | ']' | '}' if !quoted => {
                depth -= 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced brackets in {line}")
}

/// Splits at `separator` outside brackets and string literals, trimming each part.
fn split_top_level(text: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0_usize;
    let mut quoted = false;
    let mut start = 0;
    for (index, character) in text.char_indices() {
        match character {
            '"' => quoted = !quoted,
            '(' | '[' | '{' if !quoted => depth += 1,
            ')' | ']' | '}' if !quoted => depth = depth.saturating_sub(1),
            _ if character == separator && depth == 0 && !quoted => {
                parts.push(text[start..index].trim());
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim());
    parts.retain(|part| !part.is_empty());
    parts
}

/// Requires both renderers to admit the module, checks the MLIR structure, and, when LLVM tools
/// are configured, requires identical optimized machine code and metadata.
#[track_caller]
pub fn assert_equivalent(name: &str, module: &Module, target: MlirTarget) -> String {
    let caller = std::panic::Location::caller().file();
    let text = text(module, target)
        .unwrap_or_else(|error| panic!("{name}: textual lowering rejected the module: {error:?}"));
    let mlir = mlir(module, target)
        .unwrap_or_else(|error| panic!("{name}: MLIR lowering rejected the module: {error:?}"));
    assert_structure(name, &text, &mlir, target);
    if std::env::var_os("FE2O3_MLIR_DUMP_DIR").is_some() {
        let dumped = with_wave64_contract(module.clone());
        let text = self::text(&dumped, target).unwrap();
        let mlir = self::mlir(&dumped, target).unwrap();
        dump_pair(caller, name, &text, &mlir, target);
    }
    match llvm_tools() {
        Some(tools) => differential(&tools, name, &text, &mlir, target),
        None => differential_unavailable(name),
    }
    mlir
}

/// Without `FE2O3_LLVM_BIN` only the LLVM-free checks run. Says so once per test binary on the
/// real stderr (libtest captures `eprintln!` of passing tests), or fails when
/// `FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1`.
fn differential_unavailable(name: &str) {
    static NOTICE: Once = Once::new();
    assert!(
        std::env::var_os("FE2O3_REQUIRE_MLIR_DIFFERENTIAL").is_none_or(|value| value != "1"),
        "{name}: FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1 but FE2O3_LLVM_BIN is unset"
    );
    NOTICE.call_once(|| {
        let _ = writeln!(
            std::io::stderr(),
            "note: FE2O3_LLVM_BIN is unset; the MLIR differential is skipped and only structural \
             checks run (set FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1 to fail instead)"
        );
    });
}

/// Requires the textual renderer to admit the module while the MLIR renderer fails closed.
pub fn assert_fails_closed(name: &str, module: &Module, target: MlirTarget, needle: &str) {
    text(module, target)
        .unwrap_or_else(|error| panic!("{name}: textual lowering rejected the module: {error:?}"));
    let error = mlir(module, target).expect_err(name);
    assert!(
        error.diagnostics().iter().all(|diagnostic| matches!(
            diagnostic.code,
            LoweringDiagnosticCode::UnsupportedOperation | LoweringDiagnosticCode::UnsupportedType
        ) && diagnostic
            .message
            .contains("MLIR LLVM V1 emitter")),
        "{name}: unexpected rejection {error:?}"
    );
    assert!(
        error
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains(needle)),
        "{name}: rejection does not mention {needle:?}: {error:?}"
    );
}

/// The worker admits a compiler kernel only with an explicit wavefront contract
/// (`target-features`). A module that leaves the width open everywhere gets CDNA's wave64 for the
/// dumped corpus; the in-process differential keeps the module as given.
fn with_wave64_contract(mut module: Module) -> Module {
    let open = |capabilities: &BTreeSet<TargetCapability>| {
        !capabilities
            .iter()
            .any(|capability| matches!(capability, TargetCapability::WaveWidth(_)))
    };
    if open(&module.required_capabilities)
        && module
            .kernels
            .iter()
            .all(|kernel| open(&kernel.required_capabilities))
        && module
            .functions
            .iter()
            .all(|function| open(&function.required_capabilities))
    {
        module
            .required_capabilities
            .insert(TargetCapability::WaveWidth(WaveWidth::Wave64));
    }
    module
}

/// Names each pair after the calling test file so equal names from different files do not
/// collide. Writes go through a rename because tests in one binary run concurrently; a pair that
/// differs from one already dumped under the same stem fails instead of overwriting it.
fn dump_pair(caller: &str, name: &str, text: &str, mlir: &str, target: MlirTarget) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let Some(directory) = std::env::var_os("FE2O3_MLIR_DUMP_DIR").map(PathBuf::from) else {
        return;
    };
    std::fs::create_dir_all(&directory).unwrap();
    let file = Path::new(caller)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("corpus");
    let stem = format!("{file}-{name}.{}", target.cpu());
    for (extension, contents) in [("ll", text), ("mlir", mlir)] {
        let path = directory.join(format!("{stem}.{extension}"));
        if let Ok(existing) = std::fs::read_to_string(&path) {
            assert!(
                existing == contents,
                "{stem}: {} already holds a different module; empty FE2O3_MLIR_DUMP_DIR first",
                path.display()
            );
            continue;
        }
        let partial = directory.join(format!(
            ".{stem}.{extension}.{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&partial, contents).unwrap();
        std::fs::rename(&partial, &path).unwrap();
    }
}

struct LlvmTools {
    translate: PathBuf,
    opt: PathBuf,
    llc: PathBuf,
}

fn llvm_tools() -> Option<LlvmTools> {
    let directory = PathBuf::from(std::env::var_os("FE2O3_LLVM_BIN")?);
    let tools = LlvmTools {
        translate: directory.join("mlir-translate"),
        opt: directory.join("opt"),
        llc: directory.join("llc"),
    };
    for tool in [&tools.translate, &tools.opt, &tools.llc] {
        assert!(tool.is_file(), "FE2O3_LLVM_BIN lacks {}", tool.display());
    }
    Some(tools)
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var_os("FE2O3_MLIR_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn scratch(name: &str) -> Scratch {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "fe2o3-mlir-v1-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    Scratch(path)
}

fn run(name: &str, step: &str, command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{name}: {step} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn compile(tools: &LlvmTools, name: &str, input: &Path, cpu: &str) -> String {
    let optimized = input.with_extension("opt.bc");
    let assembly = input.with_extension("s");
    run(
        name,
        "opt -O3",
        Command::new(&tools.opt)
            .arg("-O3")
            .arg(input)
            .arg("-o")
            .arg(&optimized),
    );
    run(
        name,
        "llc -O3",
        Command::new(&tools.llc)
            .args(["-O3", "-mtriple=amdgcn-amd-amdhsa"])
            .arg(format!("-mcpu={cpu}"))
            .arg(&optimized)
            .arg("-o")
            .arg(&assembly),
    );
    std::fs::read_to_string(assembly).unwrap()
}

/// Keeps instructions, directives, the kernel descriptor, and HSA metadata. Drops comments,
/// file identity, and positional kernel-argument names (mlir-translate drops LLVM argument names;
/// the worker restores them from `loc("...")`, which [`assert_structure`] checks).
pub fn normalize_assembly(assembly: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut active = false;
    for line in assembly.lines() {
        let code = line.split(';').next().unwrap().trim_end();
        let trimmed = code.trim_start();
        if !active {
            active = trimmed.starts_with(".text");
        }
        if !active || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with(".file") || trimmed.starts_with(".ident") {
            continue;
        }
        let item = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        if let Some(value) = item.strip_prefix(".name:") {
            let value = value.trim();
            if value
                .strip_prefix("arg")
                .is_some_and(|rest| rest.starts_with(|character: char| character.is_ascii_digit()))
            {
                continue;
            }
        }
        lines.push(item.to_owned());
        if trimmed.starts_with(".end_amdgpu_metadata") {
            break;
        }
    }
    lines
}

/// Rewrites every `%` or `@` identifier outside string literals, plain or quoted; `rename`
/// returning `None` keeps the token.
fn rewrite_identifiers(line: &str, rename: impl FnMut(&str) -> Option<String>) -> String {
    rewrite_tokens(line, b"%@", rename)
}

/// Rewrites every token made of one of `sigils` and a plain or quoted identifier, outside string
/// literals.
fn rewrite_tokens(
    line: &str,
    sigils: &[u8],
    mut rename: impl FnMut(&str) -> Option<String>,
) -> String {
    let bytes = line.as_bytes();
    let mut output = String::with_capacity(line.len());
    let mut index = 0;
    while index < line.len() {
        let byte = bytes[index];
        let end = if byte == b'"' {
            closing_quote(line, index)
        } else if sigils.contains(&byte) {
            if bytes.get(index + 1) == Some(&b'"') {
                closing_quote(line, index + 1)
            } else {
                line[index + 1..]
                    .find(|character: char| !is_identifier(character))
                    .map_or(line.len(), |offset| index + 1 + offset)
            }
        } else {
            index
        };
        if end > index + 1 && byte != b'"' {
            let token = &line[index..end];
            output.push_str(&rename(token).unwrap_or_else(|| token.to_owned()));
            index = end;
        } else if end > index {
            output.push_str(&line[index..end]);
            index = end;
        } else {
            let character = line[index..].chars().next().unwrap();
            output.push(character);
            index += character.len_utf8();
        }
    }
    output
}

/// Index just past the string literal that opens at `open`.
fn closing_quote(line: &str, open: usize) -> usize {
    line[open + 1..]
        .find('"')
        .map_or(line.len(), |offset| open + offset + 2)
}

fn is_identifier(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '$')
}

/// The line without its `;` comment.
fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    for (index, character) in line.char_indices() {
        match character {
            '"' => quoted = !quoted,
            ';' if !quoted => return &line[..index],
            _ => {}
        }
    }
    line
}

/// The label a body line declares, if it is one.
fn label(line: &str) -> Option<&str> {
    let name = line.strip_suffix(':')?;
    let plain = !name.is_empty() && name.chars().all(is_identifier);
    let quoted = name.len() >= 2 && name.starts_with('"') && name.ends_with('"');
    (plain || quoted).then_some(name)
}

/// The global an all-zero-index `getelementptr` of a global denotes: `mlir-translate`'s
/// `IRBuilder` folds those to the global itself, the textual emitter keeps the instruction.
fn zero_global_offset(instruction: &str) -> Option<String> {
    let mut rest = instruction.strip_prefix("getelementptr ")?;
    while let Some(flagless) = ["inbounds ", "nuw ", "nusw "]
        .iter()
        .find_map(|flag| rest.strip_prefix(flag))
    {
        rest = flagless;
    }
    let parts = split_top_level(rest, ',');
    let (pointer, indices) = parts.get(1..)?.split_first()?;
    let (space, global) = pointer.rsplit_once(' ')?;
    let pointer_type = space == "ptr"
        || space
            .strip_prefix("ptr addrspace(")
            .and_then(|rest| rest.strip_suffix(')'))
            .is_some_and(|number| number.parse::<u32>().is_ok());
    (pointer_type
        && global.starts_with('@')
        && !indices.is_empty()
        && indices
            .iter()
            .all(|index| matches!(*index, "i32 0" | "i64 0")))
    .then(|| global.to_owned())
}

/// Orders `phi` incoming pairs by predecessor; the two emitters list them differently.
fn sort_phi(line: String) -> String {
    let Some(open) = line
        .find(" = phi ")
        .and_then(|at| line[at..].find(" [ ").map(|offset| at + offset + 1))
    else {
        return line;
    };
    let mut incoming = split_top_level(&line[open..], ',');
    incoming.sort_by_key(|pair| {
        pair.trim_start_matches('[')
            .trim_end_matches(']')
            .rsplit_once(',')
            .and_then(|(_, block)| block.trim().strip_prefix("%bb")?.parse::<usize>().ok())
    });
    format!("{}{}", &line[..open], incoming.join(", "))
}

/// Replaces `#N` attribute groups by their sorted attributes and `!N` metadata by its content.
fn resolve_references(
    line: &str,
    groups: &BTreeMap<String, String>,
    metadata: &BTreeMap<String, String>,
    depth: usize,
) -> String {
    rewrite_tokens(line, b"#!", |token| {
        let (sigil, key) = token.split_at(1);
        if !key.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        if sigil == "#" {
            return Some(
                groups
                    .get(key)
                    .unwrap_or_else(|| panic!("undefined {token}"))
                    .clone(),
            );
        }
        assert!(
            depth < 16,
            "metadata {token} nests too deeply or refers to itself"
        );
        let node = metadata
            .get(key)
            .unwrap_or_else(|| panic!("undefined {token}"));
        Some(resolve_references(node, groups, metadata, depth + 1))
    })
}

/// Replaces named types by their bodies; the textual emitter names helper result structs that
/// `mlir-translate` prints as literal structs.
fn inline_types(line: &str, types: &BTreeMap<String, String>, depth: usize) -> String {
    rewrite_identifiers(line, |token| {
        let body = types.get(token)?;
        assert!(depth < 16, "named type {token} nests too deeply");
        Some(inline_types(body, types, depth + 1))
    })
}

/// A form of an upgraded (bitcode round-tripped) LLVM module that is equal for the two emitters
/// exactly when they produced the same module up to spelling: named types, attribute groups, and
/// metadata nodes are inlined (attributes sorted); value, argument, and block names become
/// positional (`%aN`, `%vN`, `%bbN`); all-zero-index `getelementptr`s of globals become the
/// global; `phi` incoming pairs are sorted; the target lines, globals, and used declarations are
/// sorted; functions keep their order. Comments, the module identity, named metadata (the worker
/// owns the module flags), and unused declarations are dropped.
pub fn canonical_ir(module: &str) -> Vec<String> {
    let mut types = BTreeMap::new();
    let mut groups = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    let mut references = BTreeMap::<String, usize>::new();
    for line in module.lines() {
        let line = strip_comment(line).trim_end();
        rewrite_identifiers(line, |token| {
            *references.entry(token.to_owned()).or_default() += 1;
            None
        });
        if line.starts_with('%')
            && let Some((name, body)) = line.split_once(" = type ")
        {
            types.insert(name.to_owned(), body.to_owned());
        } else if let Some(rest) = line.strip_prefix("attributes #") {
            let (key, body) = rest.split_once(" = { ").unwrap();
            let mut attributes = split_top_level(body.strip_suffix('}').unwrap(), ' ');
            attributes.sort_unstable();
            groups.insert(key.to_owned(), format!("{{{}}}", attributes.join(" ")));
        } else if let Some(rest) = line.strip_prefix('!')
            && rest.starts_with(|character: char| character.is_ascii_digit())
        {
            let (key, body) = rest.split_once(" = ").unwrap();
            metadata.insert(key.to_owned(), body.to_owned());
        }
    }
    let canonical =
        |line: &str| inline_types(&resolve_references(line, &groups, &metadata, 0), &types, 0);
    let mut lines = module
        .lines()
        .map(|line| strip_comment(line).trim_end())
        .filter(|line| {
            line.starts_with("target ")
                || line.starts_with('@')
                || (line.starts_with("declare ")
                    && references[&format!("@{}", signature(line).0)] > 1)
        })
        .map(canonical)
        .collect::<Vec<_>>();
    lines.sort_unstable();
    let mut source = module.lines().map(|line| strip_comment(line).trim());
    while let Some(line) = source.next() {
        let Some(header) = line.strip_prefix("define ") else {
            continue;
        };
        let header = header.strip_suffix('{').unwrap().trim_end();
        let body = source
            .by_ref()
            .take_while(|line| *line != "}")
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        let mut names = BTreeMap::new();
        let mut numbered = 0;
        for (index, parameter) in split_top_level(signature(header).1, ',')
            .into_iter()
            .enumerate()
        {
            let mut last = None;
            rewrite_identifiers(parameter, |token| {
                last = token
                    .starts_with('%')
                    .then(|| token.to_owned())
                    .or(last.take());
                None
            });
            let parameter = last.unwrap_or_else(|| panic!("unnamed parameter in define {header}"));
            numbered += usize::from(parameter[1..].bytes().all(|byte| byte.is_ascii_digit()));
            names.insert(parameter, format!("%a{index}"));
        }
        if body.first().is_none_or(|line| label(line).is_none()) {
            names.insert(format!("%{numbered}"), "%bb0".to_owned());
        }
        let mut instructions = Vec::new();
        let mut blocks = 0;
        let mut values = 0;
        for line in body {
            if let Some(name) = label(line) {
                if !instructions.is_empty() {
                    blocks += 1;
                    instructions.push(format!("bb{blocks}:"));
                }
                names.insert(format!("%{name}"), format!("%bb{blocks}"));
                continue;
            }
            if line.starts_with('%')
                && let Some((value, instruction)) = line.split_once(" = ")
            {
                if let Some(global) = zero_global_offset(instruction) {
                    names.insert(value.to_owned(), global);
                    continue;
                }
                names.insert(value.to_owned(), format!("%v{values}"));
                values += 1;
            }
            instructions.push(line.to_owned());
        }
        let rename = |line: &str| rewrite_identifiers(line, |token| names.get(token).cloned());
        lines.push(format!("define {}", canonical(&rename(header))));
        lines.extend(
            instructions
                .iter()
                .map(|line| format!("  {}", sort_phi(canonical(&rename(line))))),
        );
    }
    lines
}

/// Round-trips `input` through bitcode as the worker's pipeline does, so both emitters' modules
/// get the same attribute upgrades (`"denormal-fp-math-f32"` becomes `denormal_fpenv`), and
/// returns the upgraded text.
fn upgraded(tools: &LlvmTools, name: &str, input: &Path) -> String {
    let bitcode = input.with_extension("upgrade.bc");
    let text = input.with_extension("upgraded.ll");
    run(
        name,
        "opt verify",
        Command::new(&tools.opt)
            .arg("-passes=verify")
            .arg(input)
            .arg("-o")
            .arg(&bitcode),
    );
    run(
        name,
        "opt verify -S",
        Command::new(&tools.opt)
            .args(["-passes=verify", "-S"])
            .arg(&bitcode)
            .arg("-o")
            .arg(&text),
    );
    std::fs::read_to_string(text).unwrap()
}

fn assert_same_lines(
    name: &str,
    target: MlirTarget,
    what: &str,
    expected: &[String],
    actual: &[String],
    directory: &Scratch,
) {
    if expected == actual {
        return;
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or(expected.len().min(actual.len()));
    let shown = |line: Option<&String>| {
        line.map_or_else(
            || "<end>".to_owned(),
            |line| line.chars().take(300).collect::<String>(),
        )
    };
    panic!(
        "{name} ({}): {what} differs at line {first} of {}/{}\n  text: {}\n  mlir: {}\n\
         scratch {} (set FE2O3_MLIR_KEEP=1 to keep it)",
        target.cpu(),
        expected.len(),
        actual.len(),
        shown(expected.get(first)),
        shown(actual.get(first)),
        directory.0.display()
    );
}

fn differential(tools: &LlvmTools, name: &str, text: &str, mlir: &str, target: MlirTarget) {
    let directory = scratch(&format!("{name}-{}", target.cpu()));
    let text_path = directory.0.join("text.ll");
    let mlir_path = directory.0.join("module.mlir");
    let translated = directory.0.join("translated.ll");
    std::fs::write(&text_path, text).unwrap();
    std::fs::write(&mlir_path, mlir).unwrap();
    run(
        name,
        "mlir-translate",
        Command::new(&tools.translate)
            .arg("--mlir-to-llvmir")
            .arg(&mlir_path)
            .arg("-o")
            .arg(&translated),
    );
    let expected = canonical_ir(&upgraded(tools, name, &text_path));
    let actual = canonical_ir(&upgraded(tools, name, &translated));
    std::fs::write(directory.0.join("text.canon"), expected.join("\n")).unwrap();
    std::fs::write(directory.0.join("translated.canon"), actual.join("\n")).unwrap();
    assert_same_lines(name, target, "canonical IR", &expected, &actual, &directory);
    let expected = normalize_assembly(&compile(tools, name, &text_path, target.cpu()));
    let actual = normalize_assembly(&compile(tools, name, &translated, target.cpu()));
    assert!(
        expected
            .iter()
            .any(|line| line.starts_with(".amdhsa_kernel")),
        "{name}: assembly lacks a kernel descriptor"
    );
    assert_same_lines(
        name,
        target,
        "optimized assembly",
        &expected,
        &actual,
        &directory,
    );
}
