#include "WorkerMlirInput.h"
#include "WorkerProtocol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"

#if FE2O3_WORKER_MLIR_INPUT
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VirtualFileSystem.h"
#endif

#include <string>
#include <vector>

namespace fe2o3::worker {
namespace {

constexpr llvm::StringLiteral TruncationMarker = "<truncated>";
/// The pipeline prefixes the input name and then cuts each diagnostic at
/// MaxDiagnosticBytes without a marker; this stage bounds its own messages
/// below that, so a cut message still says it was cut.
constexpr size_t MaxMessageBytes = MaxDiagnosticBytes - 256;

/// Every message of this stage goes through here: refusals echo operation and
/// argument names, and parser diagnostics echo input text, so none of them
/// has a length the input cannot choose.
llvm::Error mlirInputError(const llvm::Twine &Message) {
  std::string Text = Message.str();
  if (Text.size() > MaxMessageBytes) {
    Text.resize(MaxMessageBytes - TruncationMarker.size());
    Text.append(TruncationMarker.data(), TruncationMarker.size());
  }
  return llvm::make_error<llvm::StringError>(Text,
                                             llvm::inconvertibleErrorCode());
}

#if FE2O3_WORKER_MLIR_INPUT

/// LLVM truncates longer local names (`-non-global-value-max-name-size`),
/// which would silently rename an argument.
constexpr size_t MaxArgumentNameBytes = 1024;

/// Collects MLIR diagnostics for one message. Errors are kept apart from
/// warnings and remarks and always come first, so the diagnostic that caused a
/// refusal survives the message bound; each entry carries its severity and
/// notes. Collection stops once a severity class alone exceeds the bound.
class MlirDiagnostics {
public:
  void append(mlir::Diagnostic &Value) {
    bool IsError = Value.getSeverity() == mlir::DiagnosticSeverity::Error;
    std::string &Target = IsError ? Errors : Others;
    if (Target.size() > MaxMessageBytes)
      return;
    std::string Line;
    llvm::raw_string_ostream Stream(Line);
    Stream << severity(Value.getSeverity()) << ": " << Value.getLocation()
           << ": " << Value;
    for (mlir::Diagnostic &Note : Value.getNotes())
      Stream << " (note: " << Note.getLocation() << ": " << Note << ")";
    Stream.flush();
    if (!Target.empty())
      Target += "; ";
    Target += Line;
  }

  std::string text() const {
    if (Errors.empty() || Others.empty())
      return Errors + Others;
    return Errors + "; " + Others;
  }

private:
  static llvm::StringRef severity(mlir::DiagnosticSeverity Value) {
    switch (Value) {
    case mlir::DiagnosticSeverity::Error:
      return "error";
    case mlir::DiagnosticSeverity::Warning:
      return "warning";
    case mlir::DiagnosticSeverity::Remark:
      return "remark";
    case mlir::DiagnosticSeverity::Note:
      return "note";
    }
    return "diagnostic";
  }

  std::string Errors;
  std::string Others;
};

/// Removes the `Debug Info Version` flag that translation adds unconditionally
/// when the module carries no debug compile unit, so a debug-free MLIR module
/// reaches the shared path with the same module flags as its textual IR form.
/// The compile-unit list is read by name: iterating debug_compile_units()
/// casts every operand and would trust metadata no verifier has seen yet.
void dropUnusedDebugInfoVersion(llvm::Module &ModuleValue) {
  if (llvm::NamedMDNode *Units = ModuleValue.getNamedMetadata("llvm.dbg.cu");
      Units && Units->getNumOperands() != 0)
    return;
  llvm::NamedMDNode *Flags = ModuleValue.getModuleFlagsMetadata();
  if (!Flags)
    return;
  llvm::SmallVector<llvm::MDNode *, 8> Kept;
  bool Dropped = false;
  for (llvm::MDNode *Flag : Flags->operands()) {
    auto *Key = Flag->getNumOperands() == 3
                    ? llvm::dyn_cast<llvm::MDString>(Flag->getOperand(1))
                    : nullptr;
    if (Key && Key->getString() == "Debug Info Version") {
      Dropped = true;
      continue;
    }
    Kept.push_back(Flag);
  }
  if (!Dropped)
    return;
  Flags->clearOperands();
  for (llvm::MDNode *Flag : Kept)
    Flags->addOperand(Flag);
  if (Flags->getNumOperands() == 0)
    ModuleValue.eraseNamedMetadata(Flags);
}

bool isAdmittedAlignment(uint64_t Value) {
  return llvm::isPowerOf2_64(Value) && Value <= llvm::Value::MaximumAlignment;
}

std::string describeRange(mlir::LLVM::ConstantRangeAttr Range) {
  return (llvm::Twine("<i") + llvm::Twine(Range.getLower().getBitWidth()) +
          ", " + llvm::toString(Range.getLower(), 10, true) + ", " +
          llvm::toString(Range.getUpper(), 10, true) + ">")
      .str();
}

/// Upstream translation builds llvm::ConstantRange from a range attribute
/// without checking it; equal bounds other than min/max assert there, and the
/// remaining equal-bound spellings (empty and full sets) are refused as well.
std::string refuseConstantRanges(mlir::Attribute Value) {
  std::string Refusal;
  Value.walk([&](mlir::LLVM::ConstantRangeAttr Range) {
    if (Refusal.empty() && Range.getLower() == Range.getUpper())
      Refusal = "carries the constant range " + describeRange(Range) +
                " with equal bounds";
  });
  return Refusal;
}

/// Inherent `alignment` values (llvm.func, llvm.load, llvm.store, llvm.alloca,
/// atomics) reach llvm::Align unchecked; only llvm.mlir.global verifies them.
/// Zero is the absent spelling, which translation skips.
std::string refuseInherentAlignment(mlir::Operation &Operation) {
  auto Alignment = Operation.getAttrOfType<mlir::IntegerAttr>("alignment");
  if (!Alignment)
    return {};
  llvm::APInt Value = Alignment.getValue();
  if (Value.isZero() || (Value.getActiveBits() <= 64 &&
                         isAdmittedAlignment(Value.getZExtValue())))
    return {};
  return "carries alignment " + llvm::toString(Value, 10, false) +
         ", which is not a power of two within LLVM's bound";
}

bool isUnitParameterAttribute(llvm::StringRef Name) {
  using mlir::LLVM::LLVMDialect;
  return Name == LLVMDialect::getNoAliasAttrName() ||
         Name == LLVMDialect::getNoUndefAttrName() ||
         Name == LLVMDialect::getNonNullAttrName() ||
         Name == LLVMDialect::getNoCaptureAttrName() ||
         Name == LLVMDialect::getNoFreeAttrName() ||
         Name == LLVMDialect::getReadonlyAttrName() ||
         Name == LLVMDialect::getWriteOnlyAttrName() ||
         Name == LLVMDialect::getReadnoneAttrName();
}

/// Checks one parameter attribute of the memory-fact allowlist. Translation
/// silently drops every entry it does not map (a misspelled `llvm.noalis`, an
/// unprefixed name), the ABI-changing ones (byval, sret, inreg, zeroext, ...)
/// are outside the engineering subset, and call-site entries never pass the
/// dialect's parameter verifier, so value kinds are checked here: translation
/// asserts on a mismatched kind instead of diagnosing it.
std::string refuseParameterAttribute(mlir::NamedAttribute Attribute,
                                     mlir::Type ValueType) {
  using mlir::LLVM::LLVMDialect;
  llvm::StringRef Name = Attribute.getName().strref();
  mlir::Attribute Value = Attribute.getValue();
  auto Refuse = [&](const llvm::Twine &Why) {
    return (llvm::Twine("carries parameter attribute '") + Name + "' " + Why)
        .str();
  };
  if (isUnitParameterAttribute(Name))
    return llvm::isa<mlir::UnitAttr>(Value) ? std::string()
                                            : Refuse("with a value");
  if (Name == LLVMDialect::getAlignAttrName() ||
      Name == LLVMDialect::getDereferenceableAttrName() ||
      Name == LLVMDialect::getDereferenceableOrNullAttrName()) {
    auto Integer = llvm::dyn_cast<mlir::IntegerAttr>(Value);
    if (!Integer || Integer.getValue().getActiveBits() > 64)
      return Refuse("without a 64-bit integer value");
    uint64_t Bytes = Integer.getValue().getZExtValue();
    if (Name == LLVMDialect::getAlignAttrName())
      return isAdmittedAlignment(Bytes)
                 ? std::string()
                 : Refuse("with a value that is not a power of two within "
                          "LLVM's bound");
    return Bytes != 0 ? std::string() : Refuse("with a zero byte count");
  }
  if (Name == LLVMDialect::getRangeAttrName()) {
    auto Range = llvm::dyn_cast<mlir::LLVM::ConstantRangeAttr>(Value);
    if (!Range)
      return Refuse("without a constant range value");
    auto Integer = llvm::dyn_cast_or_null<mlir::IntegerType>(ValueType);
    if (!Integer || Integer.getWidth() != Range.getLower().getBitWidth())
      return Refuse("whose width does not match its integer value");
    return {};
  }
  return Refuse("outside the admitted memory facts");
}

/// Types of the values that `arg_attrs` and `res_attrs` describe: function
/// parameters and results for a definition or declaration, call arguments and
/// results for a call site.
void parameterTypes(mlir::Operation &Operation,
                    llvm::SmallVectorImpl<mlir::Type> &Arguments,
                    llvm::SmallVectorImpl<mlir::Type> &Results) {
  if (auto Function = llvm::dyn_cast<mlir::FunctionOpInterface>(&Operation)) {
    llvm::append_range(Arguments, Function.getArgumentTypes());
    llvm::append_range(Results, Function.getResultTypes());
    return;
  }
  if (auto Call = llvm::dyn_cast<mlir::CallOpInterface>(&Operation))
    llvm::append_range(Arguments, Call.getArgOperands().getTypes());
  else
    llvm::append_range(Arguments, Operation.getOperandTypes());
  llvm::append_range(Results, Operation.getResultTypes());
}

std::string refuseParameterAttributes(mlir::Operation &Operation) {
  auto Attributed =
      llvm::dyn_cast<mlir::ArgAndResultAttrsOpInterface>(&Operation);
  if (!Attributed)
    return {};
  llvm::SmallVector<mlir::Type, 8> ArgumentTypes;
  llvm::SmallVector<mlir::Type, 2> ResultTypes;
  parameterTypes(Operation, ArgumentTypes, ResultTypes);
  for (auto [Entries, Types] :
       {std::pair{Attributed.getArgAttrsAttr(),
                  llvm::ArrayRef<mlir::Type>(ArgumentTypes)},
        std::pair{Attributed.getResAttrsAttr(),
                  llvm::ArrayRef<mlir::Type>(ResultTypes)}}) {
    if (!Entries)
      continue;
    if (Entries.size() > Types.size())
      return "carries more parameter attribute lists than values";
    for (auto [Entry, Type] : llvm::zip_first(Entries, Types)) {
      auto Dictionary = llvm::dyn_cast<mlir::DictionaryAttr>(Entry);
      if (!Dictionary)
        return "carries a non-dictionary parameter attribute list";
      for (mlir::NamedAttribute Attribute : Dictionary) {
        std::string Refusal = refuseParameterAttribute(Attribute, Type);
        if (!Refusal.empty())
          return Refusal;
      }
    }
  }
  return {};
}

/// `llvm.inline_asm` passes neither the parameter allowlist nor LLParser's
/// constraint check: translation casts `operand_attrs` entries unchecked, and
/// llvm::InlineAsm::verify runs only in LLParser. Both are settled here.
std::string refuseInlineAssembly(mlir::LLVM::InlineAsmOp Assembly) {
  if (mlir::ArrayAttr Attributes = Assembly.getOperandAttrsAttr();
      Attributes && !Attributes.empty())
    return "carries inline-assembly operand attributes";
  // A scratch context: translating a named struct here would otherwise claim
  // its name in the module context and rename the translated type.
  llvm::LLVMContext Scratch;
  mlir::LLVM::TypeToLLVMIRTranslator Types(Scratch);
  llvm::SmallVector<llvm::Type *, 8> Parameters;
  for (mlir::Type Operand : Assembly->getOperandTypes()) {
    if (!mlir::LLVM::isCompatibleType(Operand))
      return "passes a non-LLVM type to inline assembly";
    Parameters.push_back(Types.translateType(Operand));
  }
  llvm::Type *Result = llvm::Type::getVoidTy(Scratch);
  if (Assembly->getNumResults() == 1) {
    mlir::Type ResultType = Assembly->getResult(0).getType();
    if (!mlir::LLVM::isCompatibleType(ResultType))
      return "returns a non-LLVM type from inline assembly";
    Result = Types.translateType(ResultType);
  }
  llvm::FunctionType *Signature =
      llvm::FunctionType::get(Result, Parameters, /*isVarArg=*/false);
  if (llvm::Error Error =
          llvm::InlineAsm::verify(Signature, Assembly.getConstraints()))
    return "carries inline-assembly constraints that do not match its "
           "operands: " +
           llvm::toString(std::move(Error));
  return {};
}

/// Symbol names are LLVM global names; a NUL byte asserts in Value::setName.
std::string refuseSymbolName(mlir::Operation &Operation) {
  auto Name = Operation.getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (Name && Name.getValue().contains('\0'))
    return "names a symbol with a NUL byte";
  return {};
}

/// Entry-block argument names of one function definition. MLIR keeps no SSA
/// names, so a textual LLVM IR name (published as HSACO `.name` argument
/// metadata) is carried as the argument location: `%a: f32 loc("arg0")`.
struct FunctionArgumentNames {
  std::string Function;
  llvm::SmallVector<std::string, 8> Names;
};

/// Collects `loc("name")` argument names. An empty, repeated, overlong or
/// NUL-carrying name is refused, and so is a name nested inside another
/// location kind (it would silently leave the argument unnamed); a location
/// without any name leaves the argument unnamed.
std::string collectArgumentNames(mlir::LLVM::LLVMFuncOp Function,
                                 std::vector<FunctionArgumentNames> &Result) {
  if (Function.isExternal())
    return {};
  FunctionArgumentNames Entry;
  Entry.Function = Function.getSymName().str();
  llvm::StringSet<> Seen;
  bool AnyNamed = false;
  for (mlir::BlockArgument Argument : Function.getArguments()) {
    auto Describe = [&](const llvm::Twine &Why) {
      return (llvm::Twine("names argument ") +
              llvm::Twine(Argument.getArgNumber()) + " of '" + Entry.Function +
              "' " + Why)
          .str();
    };
    unsigned NameLocations = 0;
    Argument.getLoc()->walk([&](mlir::Location Location) {
      if (llvm::isa<mlir::NameLoc>(Location))
        ++NameLocations;
      return mlir::WalkResult::advance();
    });
    std::string Name;
    if (auto Location = llvm::dyn_cast<mlir::NameLoc>(Argument.getLoc())) {
      if (NameLocations != 1)
        return Describe("with nested name locations");
      Name = Location.getName().str();
      if (Name.empty())
        return Describe("with an empty name");
      if (Name.size() > MaxArgumentNameBytes)
        return Describe("with a name longer than LLVM's " +
                        llvm::Twine(MaxArgumentNameBytes) +
                        "-byte local name limit");
      if (llvm::StringRef(Name).contains('\0'))
        return Describe("with a NUL byte in its name");
      if (!Seen.insert(Name).second)
        return Describe("'" + Name + "', which is already used");
      AnyNamed = true;
    } else if (NameLocations != 0) {
      return Describe("through a name nested inside another location");
    }
    Entry.Names.push_back(std::move(Name));
  }
  if (AnyNamed)
    Result.push_back(std::move(Entry));
  return {};
}

llvm::Error applyArgumentNames(llvm::Module &ModuleValue,
                               llvm::ArrayRef<FunctionArgumentNames> Entries,
                               llvm::StringRef InputName) {
  for (const FunctionArgumentNames &Entry : Entries) {
    llvm::Function *Function = ModuleValue.getFunction(Entry.Function);
    if (!Function || Function->arg_size() != Entry.Names.size())
      return mlirInputError(llvm::Twine("MLIR input ") + InputName +
                            " lost the arguments of '" + Entry.Function +
                            "' in translation");
    for (auto [Argument, Name] :
         llvm::zip_equal(Function->args(), Entry.Names)) {
      if (Name.empty())
        continue;
      Argument.setName(Name);
      // Names are unique, NUL-free and within the length bound, so a changed
      // name can only mean LLVM uniquified it against another local value.
      if (Argument.getName() != Name)
        return mlirInputError(llvm::Twine("MLIR input ") + InputName +
                              " argument name '" + Name + "' of '" +
                              Entry.Function + "' collides in LLVM IR");
    }
  }
  return llvm::Error::success();
}

/// Checks one operation of the admitted subset; an empty answer admits it.
std::string refuseOperation(mlir::Operation &Operation, bool IsRoot,
                            std::vector<FunctionArgumentNames> &ArgumentNames) {
  llvm::StringRef OperationName = Operation.getName().getStringRef();
  llvm::StringRef Namespace = Operation.getName().getDialectNamespace();
  // Admit only the LLVM-level dialects. Higher-level dialects (gpu, arith,
  // func, ...) would need conversion passes whose results this worker does
  // not measure, and a nested builtin.module would hide a second module.
  if (Namespace != "llvm" && Namespace != "rocdl" &&
      !(IsRoot && OperationName == "builtin.module"))
    return (llvm::Twine("uses operation '") + OperationName +
            "' outside the llvm and rocdl dialects")
        .str();
  // Both dialects allow unknown operations, which then reach translation
  // without any verifier having seen them.
  if (!Operation.isRegistered())
    return (llvm::Twine("uses unregistered operation '") + OperationName + "'")
        .str();
  // Arbitrary named metadata can forge llvm.dbg.cu or llvm.module.flags
  // entries that translation and the module contract read unchecked; module
  // flags have their own typed operation.
  if (llvm::isa<mlir::LLVM::NamedMetadataOp>(Operation))
    return (llvm::Twine("uses operation '") + OperationName +
            "', which this worker does not admit")
        .str();

  // Discardable attributes are refused except the root triple and layout:
  // translation silently ignores unknown ones (a misspelled `llvm.*` module
  // attribute drops its meaning without a diagnostic), dialect amendments
  // (`rocdl.kernel`, ...) rewrite functions behind the emitted contract, and
  // the triple is cast to a string attribute without verification.
  for (mlir::NamedAttribute Attribute : Operation.getDiscardableAttrs()) {
    llvm::StringRef Name = Attribute.getName().strref();
    if (IsRoot && Name == mlir::LLVM::LLVMDialect::getTargetTripleAttrName() &&
        llvm::isa<mlir::StringAttr>(Attribute.getValue()))
      continue;
    // The LLVM dialect verifier already parsed this one as a layout string.
    if (IsRoot && Name == mlir::LLVM::LLVMDialect::getDataLayoutAttrName())
      continue;
    return (llvm::Twine("carries attribute '") + Name + "' on '" +
            OperationName + "', which this worker does not admit")
        .str();
  }

  std::string Refusal = refuseSymbolName(Operation);
  if (Refusal.empty())
    Refusal = refuseConstantRanges(Operation.getAttrDictionary());
  if (Refusal.empty())
    Refusal = refuseInherentAlignment(Operation);
  if (Refusal.empty())
    Refusal = refuseParameterAttributes(Operation);
  if (Refusal.empty())
    if (auto Assembly = llvm::dyn_cast<mlir::LLVM::InlineAsmOp>(Operation))
      Refusal = refuseInlineAssembly(Assembly);
  if (!Refusal.empty())
    return (llvm::Twine(Refusal) + " on '" + OperationName +
            "', which this worker does not admit")
        .str();
  if (auto Function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(Operation))
    return collectArgumentNames(Function, ArgumentNames);
  return {};
}

#endif

} // namespace

bool mlirLlvmTextInputSupported() { return FE2O3_WORKER_MLIR_INPUT != 0; }

llvm::Expected<std::unique_ptr<llvm::Module>>
translateMlirLlvmTextInput(llvm::StringRef Bytes, llvm::StringRef InputName,
                           llvm::LLVMContext &Context,
                           MlirLayoutAdmission AdmitLayout) {
#if FE2O3_WORKER_MLIR_INPUT
  mlir::DialectRegistry Registry;
  Registry.insert<mlir::LLVM::LLVMDialect, mlir::ROCDL::ROCDLDialect>();
  mlir::registerBuiltinDialectTranslation(Registry);
  mlir::registerLLVMDialectTranslation(Registry);
  mlir::registerROCDLDialectTranslation(Registry);
  mlir::MLIRContext MlirContext(Registry,
                                mlir::MLIRContext::Threading::DISABLED);
  MlirContext.allowUnregisteredDialects(false);
  MlirContext.printOpOnDiagnostic(false);
  MlirContext.printStackTraceOnDiagnostic(false);

  MlirDiagnostics Diagnostics;
  mlir::ScopedDiagnosticHandler Handler(&MlirContext,
                                        [&](mlir::Diagnostic &Value) {
                                          Diagnostics.append(Value);
                                          return mlir::success();
                                        });

  // The MLIR lexer relies on the NUL sentinel past the end of its buffer; the
  // request bytes carry none, so the parser reads an owned copy.
  llvm::SourceMgr Sources;
  Sources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(Bytes, InputName), llvm::SMLoc());
  mlir::ParserConfig Config(&MlirContext, /*verifyAfterParse=*/true);
  mlir::OwningOpRef<mlir::ModuleOp> Parsed =
      mlir::parseSourceFile<mlir::ModuleOp>(Sources, Config);
  if (!Parsed)
    return mlirInputError(llvm::Twine("MLIR input ") + InputName +
                          " failed to parse or verify: " + Diagnostics.text());

  std::string Refusal;
  std::vector<FunctionArgumentNames> ArgumentNames;
  mlir::Operation *Root = Parsed->getOperation();
  Root->walk([&](mlir::Operation *Operation) {
    Refusal = refuseOperation(*Operation, Operation == Root, ArgumentNames);
    return Refusal.empty() ? mlir::WalkResult::advance()
                           : mlir::WalkResult::interrupt();
  });
  if (!Refusal.empty())
    return mlirInputError(llvm::Twine("MLIR input ") + InputName + " " +
                          Refusal);

  // Admit the layout before translation, where LLParser applies it: default
  // load, store and alloca alignments are taken from the starting layout.
  if (AdmitLayout) {
    auto Attribute = [&](llvm::StringRef Name) {
      auto Value = Root->getAttrOfType<mlir::StringAttr>(Name);
      return Value ? Value.getValue() : llvm::StringRef();
    };
    std::optional<std::string> Layout = AdmitLayout(
        Attribute(mlir::LLVM::LLVMDialect::getTargetTripleAttrName()),
        Attribute(mlir::LLVM::LLVMDialect::getDataLayoutAttrName()));
    if (Layout)
      Root->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(&MlirContext, *Layout));
  }

  // Translation's own LLVM verifier writes unbounded text to stderr; the
  // shared path verifies this module like a textual one, with bounded output.
  auto EmptyFileSystem =
      llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  std::unique_ptr<llvm::Module> Translated = mlir::translateModuleToLLVMIR(
      Root, Context, InputName,
      /*disableVerification=*/true, EmptyFileSystem.get());
  if (!Translated)
    return mlirInputError(llvm::Twine("MLIR input ") + InputName +
                          " failed LLVM IR translation: " + Diagnostics.text());
  if (llvm::Error Error =
          applyArgumentNames(*Translated, ArgumentNames, InputName))
    return Error;
  dropUnusedDebugInfoVersion(*Translated);
  return Translated;
#else
  (void)Bytes;
  (void)Context;
  (void)AdmitLayout;
  return mlirInputError(llvm::Twine("MLIR input ") + InputName +
                        " requires a worker built with MLIR input support");
#endif
}

} // namespace fe2o3::worker
