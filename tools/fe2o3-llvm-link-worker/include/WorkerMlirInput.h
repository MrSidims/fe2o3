#ifndef FE2O3_LLVM_LINK_WORKER_MLIR_INPUT_H
#define FE2O3_LLVM_LINK_WORKER_MLIR_INPUT_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>

namespace fe2o3::worker {

/// True only in a worker configured with FE2O3_WORKER_ENABLE_MLIR_INPUT. Such
/// a worker carries the option in its measurement preimage, so it can never
/// share a build identity with a worker that rejects MLIR input.
bool mlirLlvmTextInputSupported();

/// The textual-IR layout callback: it sees the module's triple and layout
/// strings (empty when absent) and returns the layout the module must use.
using MlirLayoutAdmission = llvm::function_ref<std::optional<std::string>(
    llvm::StringRef Triple, llvm::StringRef Layout)>;

/// Translates one textual MLIR module restricted to the `builtin` (root module
/// only), `llvm`, and `rocdl` dialects into an LLVM module in `Context`.
///
/// This stage only produces an `llvm::Module`; layout admission, the module
/// contract, verification, optimization, codegen, and linking stay on the
/// shared LLVM-module path. Parsing runs single-threaded (the worker sandbox
/// denies thread creation) and translation sees an empty virtual filesystem.
///
/// `AdmitLayout`, when given, runs before translation and its answer replaces
/// the module layout, as LLParser applies it before the body: translation bakes
/// default load, store and alloca alignments from the layout it starts with.
///
/// Discardable attributes other than the root `llvm.target_triple` (a string)
/// and `llvm.data_layout`, and parameter attributes outside a memory-fact
/// allowlist, are refused because translation would drop them silently. Values
/// that upstream translation would assert on (malformed ranges, alignments,
/// inline-assembly constraints, NUL bytes in names) are refused before it runs.
/// An entry-block argument located at `loc("name")` gets that LLVM IR name,
/// which HSACO publishes as `.name` argument metadata.
llvm::Expected<std::unique_ptr<llvm::Module>>
translateMlirLlvmTextInput(llvm::StringRef Bytes, llvm::StringRef InputName,
                           llvm::LLVMContext &Context,
                           MlirLayoutAdmission AdmitLayout = {});

} // namespace fe2o3::worker

#endif
