#include "WorkerMlirInput.h"
#include "WorkerPipeline.h"
#include "WorkerProtocol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <pthread.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef FE2O3_LLVM_BUILD_ID
#error "FE2O3_LLVM_BUILD_ID must be supplied by CMake"
#endif
#ifndef FE2O3_WORKER_BUILD_ID
#error "FE2O3_WORKER_BUILD_ID must be supplied by CMake"
#endif

using namespace fe2o3::worker;
using namespace llvm;

namespace {

[[noreturn]] void fail(const Twine &Message) {
  errs() << "MLIR input test failed: " << Message << '\n';
  std::abort();
}

void require(bool Condition, const Twine &Message) {
  if (!Condition)
    fail(Message);
}

struct TargetStrings {
  std::string Triple;
  std::string Layout;
};

TargetStrings gfx942Strings() {
  static bool Initialized = [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    return true;
  }();
  (void)Initialized;
  Triple TripleValue("amdgcn-amd-amdhsa");
  Triple::SubArchType SubArch =
      AMDGPU::getSubArch(AMDGPU::parseArchAMDGCN("gfx942"));
  if (SubArch != Triple::NoSubArch)
    TripleValue.setArch(Triple::amdgpu, SubArch);
  std::string LookupError;
  const Target *TargetValue =
      TargetRegistry::lookupTarget(TripleValue, LookupError);
  require(TargetValue != nullptr, LookupError);
  TargetOptions Options;
  std::unique_ptr<TargetMachine> Machine(TargetValue->createTargetMachine(
      TripleValue, "gfx942", "", Options, Reloc::PIC_, CodeModel::Small,
      CodeGenOptLevel::None));
  require(Machine != nullptr, "could not create the gfx942 target machine");
  return {TripleValue.str(),
          Machine->createDataLayout().getStringRepresentation()};
}

// LLVM 22 TargetMachine spelling (no m:e, no p10-p15); LLVM 24 auto-upgrades
// it.
constexpr StringLiteral Llvm22Layout =
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-"
    "p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-"
    "v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-"
    "v2048:2048-n32:64-S32-A5-G1-ni:7:8:9";

// Two math-free kernels in textual LLVM IR and their llvm/rocdl-dialect MLIR
// spelling. `@HEADER@` is the module header variant under test.
constexpr StringLiteral KernelsText = R"(@HEADER@
define amdgpu_kernel void @fe2o3_mlir_vecadd(ptr addrspace(1) noalias noundef readonly align 4 %a, ptr addrspace(1) noalias noundef readonly align 4 %b, ptr addrspace(1) noalias noundef writeonly align 4 %c) #0 !reqd_work_group_size !1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gid = call i32 @llvm.amdgcn.workgroup.id.x()
  %base = mul i32 %gid, 256
  %index = add i32 %base, %tid
  %wide = zext i32 %index to i64
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i64 %wide
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %wide
  %pc = getelementptr inbounds float, ptr addrspace(1) %c, i64 %wide
  %va = load float, ptr addrspace(1) %pa, align 4
  %vb = load float, ptr addrspace(1) %pb, align 4
  %sum = fadd float %va, %vb
  store float %sum, ptr addrspace(1) %pc, align 4
  ret void
}

define amdgpu_kernel void @fe2o3_mlir_strided_scale(ptr addrspace(1) noalias noundef align 4 %data, float %alpha, i32 %n) #0 !reqd_work_group_size !1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %start = icmp ult i32 %tid, %n
  br i1 %start, label %loop, label %exit

loop:
  %i = phi i32 [ %tid, %entry ], [ %next, %loop ]
  %wide = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %data, i64 %wide
  %v = load float, ptr addrspace(1) %p, align 4
  %m = fmul contract float %v, %alpha
  store float %m, ptr addrspace(1) %p, align 4
  %next = add nuw i32 %i, 256
  %more = icmp ult i32 %next, %n
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "target-cpu"="gfx942" "target-features"="-wavefrontsize32,+wavefrontsize64" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 256, i32 1, i32 1}
)";

constexpr StringLiteral KernelsBody = R"(
  llvm.module_flags [#llvm.mlir.module_flag<error, "amdhsa_code_object_version", 500 : i32>]
  llvm.func amdgpu_kernelcc @fe2o3_mlir_vecadd(%a: !llvm.ptr<1> {llvm.noalias, llvm.noundef, llvm.readonly, llvm.align = 4 : i64} loc("a"), %b: !llvm.ptr<1> {llvm.noalias, llvm.noundef, llvm.readonly, llvm.align = 4 : i64} loc("b"), %c: !llvm.ptr<1> {llvm.noalias, llvm.noundef, llvm.writeonly, llvm.align = 4 : i64} loc("c")) attributes {reqd_work_group_size = array<i32: 256, 1, 1>, target_cpu = "gfx942", target_features = #llvm.target_features<["-wavefrontsize32", "+wavefrontsize64"]>, passthrough = [["amdgpu-flat-work-group-size", "256,256"]]} {
    %tid = rocdl.workitem.id.x : i32
    %gid = rocdl.workgroup.id.x : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %base = llvm.mul %gid, %c256 : i32
    %index = llvm.add %base, %tid : i32
    %wide = llvm.zext %index : i32 to i64
    %pa = llvm.getelementptr inbounds %a[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %pb = llvm.getelementptr inbounds %b[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %pc = llvm.getelementptr inbounds %c[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %va = llvm.load %pa <alignment = 4> : !llvm.ptr<1> -> f32
    %vb = llvm.load %pb <alignment = 4> : !llvm.ptr<1> -> f32
    %sum = llvm.fadd %va, %vb : f32
    llvm.store %sum, %pc <alignment = 4> : f32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func amdgpu_kernelcc @fe2o3_mlir_strided_scale(%data: !llvm.ptr<1> {llvm.noalias, llvm.noundef, llvm.align = 4 : i64} loc("data"), %alpha: f32 loc("alpha"), %n: i32 loc("n")) attributes {reqd_work_group_size = array<i32: 256, 1, 1>, target_cpu = "gfx942", target_features = #llvm.target_features<["-wavefrontsize32", "+wavefrontsize64"]>, passthrough = [["amdgpu-flat-work-group-size", "256,256"]]} {
    %tid = rocdl.workitem.id.x : i32
    %start = llvm.icmp "ult" %tid, %n : i32
    llvm.cond_br %start, ^loop(%tid : i32), ^exit
  ^loop(%i: i32):
    %wide = llvm.zext %i : i32 to i64
    %p = llvm.getelementptr inbounds %data[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %v = llvm.load %p <alignment = 4> : !llvm.ptr<1> -> f32
    %m = llvm.fmul %v, %alpha fastmath<contract> : f32
    llvm.store %m, %p <alignment = 4> : f32, !llvm.ptr<1>
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %next = llvm.add %i, %c256 overflow<nuw> : i32
    %more = llvm.icmp "ult" %next, %n : i32
    llvm.cond_br %more, ^loop(%next : i32), ^exit
  ^exit:
    llvm.return
  }
)";

const std::vector<std::string> KernelExports = {"fe2o3_mlir_strided_scale",
                                                "fe2o3_mlir_vecadd"};
const std::vector<std::string> KernelFinals = {
    "fe2o3_mlir_strided_scale", "fe2o3_mlir_strided_scale.kd",
    "fe2o3_mlir_vecadd", "fe2o3_mlir_vecadd.kd"};

// Two kernels whose loads and stores carry no alignment, so LLVM takes it
// from the starting layout: i64 and LDS pointers align to 8 and 4 bytes on
// AMDGPU but to 4 and 8 bytes under the default layout, which changes the
// LDS access (ds_read_b64 or ds_read2_b32). The second kernel also reaches
// an internal LDS variable.
constexpr StringLiteral LdsText = R"(@HEADER@
@fe2o3_lds = internal addrspace(3) global [512 x i64] poison, align 8

define amdgpu_kernel void @fe2o3_mlir_lds_wide(ptr addrspace(1) noalias noundef %slots, ptr addrspace(1) noalias noundef %out) #0 !reqd_work_group_size !1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %wide = zext i32 %tid to i64
  %slotp = getelementptr inbounds ptr addrspace(3), ptr addrspace(1) %slots, i64 %wide
  %p = load ptr addrspace(3), ptr addrspace(1) %slotp
  %v = load i64, ptr addrspace(3) %p
  %dst = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %wide
  store i64 %v, ptr addrspace(1) %dst
  ret void
}

define amdgpu_kernel void @fe2o3_mlir_lds_pointer(ptr addrspace(1) noalias noundef %in, ptr addrspace(1) noalias noundef %out) #0 !reqd_work_group_size !1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %wide = zext i32 %tid to i64
  %src = getelementptr inbounds i64, ptr addrspace(1) %in, i64 %wide
  %v = load i64, ptr addrspace(1) %src
  %slot = getelementptr inbounds [512 x i64], ptr addrspace(3) @fe2o3_lds, i64 0, i64 %wide
  store i64 %v, ptr addrspace(3) %slot
  %dst = getelementptr inbounds ptr addrspace(3), ptr addrspace(1) %out, i64 %wide
  store ptr addrspace(3) %slot, ptr addrspace(1) %dst
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "target-cpu"="gfx942" "target-features"="-wavefrontsize32,+wavefrontsize64" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 256, i32 1, i32 1}
)";

constexpr StringLiteral LdsBody = R"(
  llvm.module_flags [#llvm.mlir.module_flag<error, "amdhsa_code_object_version", 500 : i32>]
  llvm.mlir.global internal @fe2o3_lds() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<512 x i64> {
    %0 = llvm.mlir.poison : !llvm.array<512 x i64>
    llvm.return %0 : !llvm.array<512 x i64>
  }
  llvm.func amdgpu_kernelcc @fe2o3_mlir_lds_wide(%slots: !llvm.ptr<1> {llvm.noalias, llvm.noundef} loc("slots"), %out: !llvm.ptr<1> {llvm.noalias, llvm.noundef} loc("out")) attributes {reqd_work_group_size = array<i32: 256, 1, 1>, target_cpu = "gfx942", target_features = #llvm.target_features<["-wavefrontsize32", "+wavefrontsize64"]>, passthrough = [["amdgpu-flat-work-group-size", "256,256"]]} {
    %tid = rocdl.workitem.id.x : i32
    %wide = llvm.zext %tid : i32 to i64
    %slotp = llvm.getelementptr inbounds %slots[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.ptr<3>
    %p = llvm.load %slotp : !llvm.ptr<1> -> !llvm.ptr<3>
    %v = llvm.load %p : !llvm.ptr<3> -> i64
    %dst = llvm.getelementptr inbounds %out[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    llvm.store %v, %dst : i64, !llvm.ptr<1>
    llvm.return
  }
  llvm.func amdgpu_kernelcc @fe2o3_mlir_lds_pointer(%in: !llvm.ptr<1> {llvm.noalias, llvm.noundef} loc("in"), %out: !llvm.ptr<1> {llvm.noalias, llvm.noundef} loc("out")) attributes {reqd_work_group_size = array<i32: 256, 1, 1>, target_cpu = "gfx942", target_features = #llvm.target_features<["-wavefrontsize32", "+wavefrontsize64"]>, passthrough = [["amdgpu-flat-work-group-size", "256,256"]]} {
    %tid = rocdl.workitem.id.x : i32
    %wide = llvm.zext %tid : i32 to i64
    %src = llvm.getelementptr inbounds %in[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %v = llvm.load %src : !llvm.ptr<1> -> i64
    %lds = llvm.mlir.addressof @fe2o3_lds : !llvm.ptr<3>
    %zero = llvm.mlir.constant(0 : i64) : i64
    %slot = llvm.getelementptr inbounds %lds[%zero, %wide] : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>, !llvm.array<512 x i64>
    llvm.store %v, %slot : i64, !llvm.ptr<3>
    %dst = llvm.getelementptr inbounds %out[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.ptr<3>
    llvm.store %slot, %dst : !llvm.ptr<3>, !llvm.ptr<1>
    llvm.return
  }
)";

const std::vector<std::string> LdsExports = {"fe2o3_mlir_lds_pointer",
                                             "fe2o3_mlir_lds_wide"};
const std::vector<std::string> LdsFinals = {
    "fe2o3_mlir_lds_pointer", "fe2o3_mlir_lds_pointer.kd",
    "fe2o3_mlir_lds_wide", "fe2o3_mlir_lds_wide.kd"};

enum class Header { Machine, Absent, Legacy };

std::string replaceAll(std::string Value, StringRef From, StringRef To) {
  size_t Position = 0;
  while ((Position = Value.find(From.str(), Position)) != std::string::npos) {
    Value.replace(Position, From.size(), To.str());
    Position += To.size();
  }
  return Value;
}

std::string textModule(StringRef Template, Header Variant) {
  TargetStrings Strings = gfx942Strings();
  std::string HeaderText;
  if (Variant == Header::Machine)
    HeaderText = "target datalayout = \"" + Strings.Layout +
                 "\"\ntarget triple = \"" + Strings.Triple + "\"\n";
  else if (Variant == Header::Legacy)
    HeaderText = "target datalayout = \"" + Llvm22Layout.str() +
                 "\"\ntarget triple = \"amdgcn-amd-amdhsa\"\n";
  return replaceAll(Template.str(), "@HEADER@", HeaderText);
}

std::string textKernels(Header Variant) {
  return textModule(KernelsText, Variant);
}

std::string mlirModule(StringRef Attributes, StringRef Body) {
  std::string Result = "module ";
  if (!Attributes.empty())
    Result += "attributes {" + Attributes.str() + "} ";
  Result += "{" + Body.str() + "}\n";
  return Result;
}

std::string machineAttributes() {
  TargetStrings Strings = gfx942Strings();
  return "llvm.data_layout = \"" + Strings.Layout +
         "\", llvm.target_triple = \"" + Strings.Triple + "\"";
}

std::string mlirHeaderAttributes(Header Variant) {
  std::string Attributes;
  if (Variant == Header::Machine)
    Attributes = machineAttributes();
  else if (Variant == Header::Legacy)
    Attributes = "llvm.data_layout = \"" + Llvm22Layout.str() +
                 "\", llvm.target_triple = \"amdgcn-amd-amdhsa\"";
  return Attributes;
}

std::string mlirKernels(Header Variant) {
  return mlirModule(mlirHeaderAttributes(Variant), KernelsBody);
}

std::vector<uint8_t> bytesOf(StringRef Text) {
  return std::vector<uint8_t>(Text.begin(), Text.end());
}

Input makeInput(InputKind Kind, std::vector<uint8_t> Bytes) {
  std::array<uint8_t, 32> Digest = SHA256::hash(Bytes);
  return {Kind, Digest, std::move(Bytes)};
}

Request makeV2Request(Input CompilerModule, std::vector<std::string> Exports,
                      std::vector<std::string> FinalSymbols,
                      StringRef Target = "gfx942:xnack-",
                      uint8_t CodeObjectVersion = 5) {
  Request Result;
  Result.Protocol = ProtocolVersion::V2;
  Result.RequestId.fill(0x31);
  Result.Identity.fill(0x72);
  Result.LlvmBuildIdentity = FE2O3_LLVM_BUILD_ID;
  Result.WorkerBuildIdentity = FE2O3_WORKER_BUILD_ID;
  Result.WorkerExecutableDigest.fill(0x51);
  Result.WorkerExecutableBytes = 4096;
  Result.CompilerEnvelopeIdentity.fill(0x62);
  Result.Target = Target.str();
  Result.CodeObjectVersion = CodeObjectVersion;
  Result.LinkOptions = {OptimizationLevel::O3, true, true};
  Result.Inputs = {CompilerModule};
  Result.CompilerModule = std::move(CompilerModule);
  llvm::sort(Exports);
  llvm::sort(FinalSymbols);
  Result.ExportSymbols = std::move(Exports);
  Result.RequiredSymbols = FinalSymbols;
  Result.ExpectedDefinedSymbols = FinalSymbols;
  Result.FinalSymbols = std::move(FinalSymbols);
  Result.MaxOutputBytes = 4 * 1024 * 1024;
  return Result;
}

Request kernelRequest(InputKind Kind, StringRef Text,
                      StringRef Target = "gfx942:xnack-") {
  return makeV2Request(makeInput(Kind, bytesOf(Text)), KernelExports,
                       KernelFinals, Target);
}

void printDiagnostics(const Response &Value) {
  for (const std::string &Diagnostic : Value.Diagnostics)
    errs() << "  diagnostic: " << Diagnostic << '\n';
}

std::vector<uint8_t> requireSuccess(const Request &RequestValue,
                                    StringRef Label, unsigned Kernels = 2) {
  Response Result = execute(RequestValue);
  if (!Result.LinkedOutput) {
    errs() << Label << " failed at stage "
           << static_cast<unsigned>(Result.FailureStage) << '\n';
    printDiagnostics(Result);
    fail(Twine(Label) + " was expected to succeed");
  }
  require(Result.FailureStage == Stage::Complete,
          Twine(Label) + " reported a stage other than Complete");
  require(Result.LinkedOutput->Digest ==
              SHA256::hash(Result.LinkedOutput->Bytes),
          Twine(Label) + " output digest is incorrect");
  require(Result.Derivation.has_value(),
          Twine(Label) + " omitted derivation evidence");
  std::string Expected =
      "post_link.check=metadata status=ok kernels=" + std::to_string(Kernels);
  bool SawMetadata =
      llvm::any_of(Result.Diagnostics, [&Expected](const std::string &D) {
        return StringRef(D).contains(Expected);
      });
  if (!SawMetadata) {
    printDiagnostics(Result);
    fail(Twine(Label) + " did not publish metadata for " + Twine(Kernels) +
         " kernels");
  }
  return Result.LinkedOutput->Bytes;
}

/// Both texts must appear in one diagnostic; `AlsoExpected` may be empty.
void requireRefusal(const Request &RequestValue, Stage ExpectedStage,
                    StringRef ExpectedText, StringRef Label,
                    StringRef AlsoExpected = "") {
  Response Result = execute(RequestValue);
  if (Result.LinkedOutput)
    fail(Twine(Label) + " was accepted");
  bool Matched = llvm::any_of(Result.Diagnostics, [&](const std::string &D) {
    return StringRef(D).contains(ExpectedText) &&
           StringRef(D).contains(AlsoExpected);
  });
  if (Result.FailureStage != ExpectedStage || !Matched) {
    errs() << Label << ": stage " << static_cast<unsigned>(Result.FailureStage)
           << " (expected " << static_cast<unsigned>(ExpectedStage)
           << "), expected text '" << ExpectedText << "'";
    if (!AlsoExpected.empty())
      errs() << " and '" << AlsoExpected << "'";
    errs() << '\n';
    printDiagnostics(Result);
    fail(Twine(Label) + " was refused for a different reason");
  }
  size_t Total = 0;
  for (const std::string &Diagnostic : Result.Diagnostics) {
    require(Diagnostic.size() <= MaxDiagnosticBytes,
            Twine(Label) + " diagnostic exceeded its byte bound");
    Total += Diagnostic.size();
  }
  require(Total <= MaxTotalDiagnosticBytes,
          Twine(Label) + " diagnostics exceeded their total byte bound");
}

void requireSameBytes(ArrayRef<uint8_t> Left, ArrayRef<uint8_t> Right,
                      StringRef Label) {
  if (Left == Right)
    return;
  size_t Limit = std::min(Left.size(), Right.size());
  size_t First = 0;
  while (First < Limit && Left[First] == Right[First])
    ++First;
  errs() << Label << ": sizes " << Left.size() << " and " << Right.size()
         << ", first difference at byte " << First << '\n';
  // Set FE2O3_MLIR_TEST_DUMP_DIR to keep both objects for llvm-readelf/objdump.
  if (const char *Directory = std::getenv("FE2O3_MLIR_TEST_DUMP_DIR")) {
    for (auto [Suffix, Bytes] : {std::pair{"left", Left}, {"right", Right}}) {
      std::string Path =
          std::string(Directory) + "/mismatch-" + Suffix + ".hsaco";
      std::error_code Error;
      raw_fd_ostream Stream(Path, Error);
      if (!Error)
        Stream.write(reinterpret_cast<const char *>(Bytes.data()),
                     Bytes.size());
    }
  }
  fail(Twine(Label) + " produced different HSACO bytes");
}

// The V2 Rust golden from CodecTests: its compiler module is LLVM bitcode.
constexpr StringLiteral V2GoldenHex =
    "46334c52455130320100200000001111111111111111111111111111111111111111"
    "1111111111111111111111110200070000006c6c766d2d7632030009000000776f"
    "726b65722d7632040028000000222222222222222222222222222222222222222222"
    "2222222222222222222222d20400000000000005000d0000006766783934323a786e61"
    "636b2d06000100000005070003000000020101080020000000333333333333333333"
    "3333333333333333333333333333333333333333333333090038000000012395170d"
    "a16ee85f3dd5dc51e591b796926a189d5f9da6bd8c3b9ef2658b71630f0000000000"
    "0000636f6d70696c65722d6d6f64756c650a003c0000000100000002eefaf5106d05"
    "5bc7739d91015ae3c8468e852ca1cf7455e178a79f70649cbfb90f00000000000000"
    "70726f76696465722d6f626a6563740b0017000000010000000f0000006578746572"
    "6e616c5f68656c7065720c0014000000010000000c0000006b65726e656c5f656e74"
    "72790d0027000000020000000f00000065787465726e616c5f68656c7065720c0000"
    "006b65726e656c5f656e7472790e000800000000100000000000000f0020000000f0"
    "06c4b5601d316904877004ab1b886057bc92ed7a0e3cf8c2b40866bcae5da4";

std::vector<uint8_t> decodeGoldenHex(StringRef Hex) {
  std::vector<uint8_t> Result;
  for (size_t I = 0; I + 1 < Hex.size(); I += 2) {
    unsigned Value = 0;
    require(!Hex.substr(I, 2).getAsInteger(16, Value), "invalid golden hex");
    Result.push_back(static_cast<uint8_t>(Value));
  }
  return Result;
}

// Re-tags the first input of one golden field as kind 4 and re-seals the
// request identity, so only the input-kind decision differs from the accepted
// golden. `FieldPrefix` ends just before that input's kind byte.
std::vector<uint8_t> goldenWithMlirInput(ArrayRef<uint8_t> FieldPrefix,
                                         InputKind Previous) {
  std::vector<uint8_t> Bytes = decodeGoldenHex(V2GoldenHex);
  auto Found = std::search(Bytes.begin(), Bytes.end(), FieldPrefix.begin(),
                           FieldPrefix.end());
  require(Found != Bytes.end(), "golden input field moved");
  auto Kind = Found + FieldPrefix.size();
  require(*Kind == static_cast<uint8_t>(Previous), "golden input kind moved");
  *Kind = static_cast<uint8_t>(InputKind::MlirLlvmText);
  const size_t IdentityOffset = Bytes.size() - 6 - 32;
  require(Bytes[IdentityOffset] == 0x0f && Bytes[IdentityOffset + 1] == 0x00,
          "golden identity field moved");
  static constexpr char Domain[] = "FE2O3/DIRECT-LLVM-WORKER-REQUEST/V2\0";
  SHA256 Hasher;
  Hasher.update(StringRef(Domain, sizeof(Domain) - 1));
  uint8_t Length[8];
  support::endian::write64le(Length, IdentityOffset);
  Hasher.update(ArrayRef<uint8_t>(Length));
  Hasher.update(ArrayRef<uint8_t>(Bytes).take_front(IdentityOffset));
  std::array<uint8_t, 32> Identity = Hasher.final();
  std::copy(Identity.begin(), Identity.end(), Bytes.end() - 32);
  return Bytes;
}

std::vector<uint8_t> goldenWithMlirCompilerModule() {
  static constexpr uint8_t ModuleField[] = {0x09, 0x00, 0x38, 0x00, 0x00, 0x00};
  return goldenWithMlirInput(ModuleField, InputKind::LlvmBitcode);
}

std::vector<uint8_t> goldenWithMlirProvider() {
  // Field 10 is the provider list; its count (1) precedes the first kind.
  static constexpr uint8_t ProviderField[] = {0x0a, 0x00, 0x3c, 0x00, 0x00,
                                              0x00, 0x01, 0x00, 0x00, 0x00};
  return goldenWithMlirInput(ProviderField, InputKind::AmdGpuRelocatable);
}

void appendLe(std::vector<uint8_t> &Out, uint64_t Value, unsigned Bytes) {
  for (unsigned I = 0; I < Bytes; ++I)
    Out.push_back(static_cast<uint8_t>(Value >> (8 * I)));
}

void appendField(std::vector<uint8_t> &Out, uint16_t Tag,
                 ArrayRef<uint8_t> Bytes) {
  appendLe(Out, Tag, 2);
  appendLe(Out, Bytes.size(), 4);
  Out.insert(Out.end(), Bytes.begin(), Bytes.end());
}

// A V1 request through its input list (fields 1 to 6). Decoding a well-formed
// input then stops at the missing required-symbol field, so a kind-4 input is
// refused only if the input decision comes first.
std::vector<uint8_t> v1RequestThroughInputs(InputKind Kind) {
  std::vector<uint8_t> Out = {'F', '3', 'L', 'R', 'E', 'Q', '0', '1'};
  appendField(Out, 1, std::vector<uint8_t>(32, 0x31));
  appendField(Out, 2, bytesOf(FE2O3_LLVM_BUILD_ID));
  appendField(Out, 3, bytesOf("gfx942:xnack-"));
  appendField(Out, 4, std::vector<uint8_t>{5});
  appendField(Out, 5, std::vector<uint8_t>{3, 1, 1});
  std::vector<uint8_t> Payload = bytesOf("module {}\n");
  std::array<uint8_t, 32> Digest = SHA256::hash(Payload);
  std::vector<uint8_t> Inputs;
  appendLe(Inputs, 1, 4);
  Inputs.push_back(static_cast<uint8_t>(Kind));
  Inputs.insert(Inputs.end(), Digest.begin(), Digest.end());
  appendLe(Inputs, Payload.size(), 8);
  Inputs.insert(Inputs.end(), Payload.begin(), Payload.end());
  appendField(Out, 6, Inputs);
  return Out;
}

void requireDecodeRefusal(Expected<Request> Decoded, StringRef ExpectedText,
                          StringRef Label) {
  if (Decoded)
    fail(Twine(Label) + " decoded");
  std::string Message = toString(Decoded.takeError());
  require(StringRef(Message).contains(ExpectedText),
          Twine(Label) + " was refused for a different reason: " + Message);
}

void testDecodeGate() {
  std::vector<uint8_t> Golden = decodeGoldenHex(V2GoldenHex);
  auto Accepted = decodeRequestV2(Golden);
  if (!Accepted)
    fail(Twine("V2 golden no longer decodes: ") +
         toString(Accepted.takeError()));
  auto Decoded = decodeRequestV2(goldenWithMlirCompilerModule());
  // MLIR is an engineering spelling of the V2 compiler module only: never a
  // provider and never a V1 input.
  StringRef SlotRefusal =
      mlirLlvmTextInputSupported()
          ? "MLIR input is admitted only as the V2 compiler module"
          : "unknown worker input kind";
  requireDecodeRefusal(decodeRequestV2(goldenWithMlirProvider()), SlotRefusal,
                       "kind-4 V2 provider");
  requireDecodeRefusal(
      decodeRequest(v1RequestThroughInputs(InputKind::MlirLlvmText)),
      SlotRefusal, "kind-4 V1 input");
  auto Control = decodeRequest(v1RequestThroughInputs(InputKind::LlvmTextIr));
  if (Control)
    fail("the truncated V1 control request decoded");
  std::string ControlMessage = toString(Control.takeError());
  require(!StringRef(ControlMessage).contains("input kind") &&
              !StringRef(ControlMessage).contains("MLIR"),
          Twine("the V1 control request was refused for its input: ") +
              ControlMessage);
  if (mlirLlvmTextInputSupported()) {
    if (!Decoded)
      fail(Twine("MLIR worker refused a kind-4 compiler module: ") +
           toString(Decoded.takeError()));
    require(Decoded->CompilerModule.Kind == InputKind::MlirLlvmText,
            "kind-4 compiler module decoded as another kind");
    return;
  }
  requireDecodeRefusal(std::move(Decoded), "unknown worker input kind",
                       "kind-4 compiler module without MLIR support");
}

void testRefusedWithoutSupport() {
  requireRefusal(
      kernelRequest(InputKind::MlirLlvmText, mlirKernels(Header::Machine)),
      Stage::InputValidation, "is MLIR, which this worker build refuses",
      "MLIR input without support");
  LLVMContext Context;
  auto Translated = translateMlirLlvmTextInput("module {}", "probe", Context);
  require(!Translated, "translation succeeded without MLIR support");
  std::string Message = toString(Translated.takeError());
  require(StringRef(Message).contains("requires a worker built with MLIR"),
          Twine("unexpected unsupported-translation message: ") + Message);
}

void testDirectTranslation() {
  LLVMContext Context;
  auto Translated = translateMlirLlvmTextInput(mlirKernels(Header::Machine),
                                               "direct", Context);
  if (!Translated)
    fail(Twine("direct translation failed: ") +
         toString(Translated.takeError()));
  Module &Value = **Translated;
  require(Value.getModuleFlag("Debug Info Version") == nullptr,
          "debug-free MLIR kept the translation Debug Info Version flag");
  auto *CodeObject = mdconst::dyn_extract_or_null<ConstantInt>(
      Value.getModuleFlag("amdhsa_code_object_version"));
  require(CodeObject && CodeObject->getZExtValue() == 500,
          "translation lost the code-object module flag");
  require(Value.getTargetTriple().str() == gfx942Strings().Triple,
          "translation changed the target triple");
  for (StringRef Name : {"fe2o3_mlir_vecadd", "fe2o3_mlir_strided_scale"}) {
    Function *Kernel = Value.getFunction(Name);
    require(Kernel != nullptr, Twine("translation lost kernel ") + Name);
    require(Kernel->getCallingConv() == CallingConv::AMDGPU_KERNEL,
            Twine(Name) + " is not an AMDGPU kernel");
    require(Kernel->getFnAttribute("amdgpu-flat-work-group-size")
                    .getValueAsString() == "256,256",
            Twine(Name) + " lost its flat workgroup bound");
    require(Kernel->getFnAttribute("target-cpu").getValueAsString() == "gfx942",
            Twine(Name) + " lost its target CPU");
    require(Kernel->getMetadata("reqd_work_group_size") != nullptr,
            Twine(Name) + " lost its required workgroup size");
  }
  Function *Scale = Value.getFunction("fe2o3_mlir_strided_scale");
  require(Scale->getArg(0)->getName() == "data" &&
              Scale->getArg(1)->getName() == "alpha" &&
              Scale->getArg(2)->getName() == "n",
          "translation did not name arguments from their locations");
}

// Forms the refusal checks must keep admitting, with their translation.
void testAdmittedForms() {
  const std::string LongName(1024, 'a');
  const std::string Body = replaceAll(R"(
  llvm.func @fe2o3_mlir_admitted(%p: !llvm.ptr<1> {llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64} loc("p"), %n: i32 {llvm.range = #llvm.constant_range<i32, 0, 100>} loc("@LONG@")) -> i32 {
    %tid = rocdl.workitem.id.x range <i32, 0, 256> : i32
    %v = llvm.inline_asm "v_mov_b32 $0, 0", "=v" : () -> i32
    llvm.inline_asm has_side_effects "s_nop 0", "" : () -> ()
    %s = llvm.add %tid, %v : i32
    %t = llvm.add %s, %n : i32
    llvm.return %t : i32
  }
)",
                                      "@LONG@", LongName);
  LLVMContext Context;
  auto Translated = translateMlirLlvmTextInput(
      mlirModule(machineAttributes(), Body), "admitted", Context);
  if (!Translated)
    fail(Twine("admitted forms were refused: ") +
         toString(Translated.takeError()));
  Function *Value = (*Translated)->getFunction("fe2o3_mlir_admitted");
  require(Value != nullptr, "translation lost the admitted-forms function");
  require(Value->getParamAlign(0) == MaybeAlign(16),
          "translation lost the parameter alignment");
  require(Value->getParamDereferenceableBytes(0) == 64,
          "translation lost the dereferenceable byte count");
  Attribute Range = Value->getParamAttribute(1, Attribute::Range);
  require(Range.isValid() &&
              Range.getRange() == ConstantRange(APInt(32, 0), APInt(32, 100)),
          "translation lost the parameter range");
  require(Value->getArg(1)->getName() == LongName,
          "translation did not keep a 1024-byte argument name");
  unsigned Assembly = 0;
  bool SawRange = false;
  for (Instruction &I : instructions(*Value)) {
    auto *Call = dyn_cast<CallInst>(&I);
    if (!Call)
      continue;
    if (Call->isInlineAsm()) {
      ++Assembly;
      continue;
    }
    if (Call->getCalledFunction() &&
        Call->getCalledFunction()->getName() == "llvm.amdgcn.workitem.id.x") {
      Attribute Result = Call->getRetAttr(Attribute::Range);
      SawRange =
          Result.isValid() &&
          Result.getRange() == ConstantRange(APInt(32, 0), APInt(32, 256));
    }
  }
  require(Assembly == 2, "translation lost an inline-assembly call");
  require(SawRange, "translation lost the work-item range");
}

std::vector<uint64_t> memoryAlignments(const Module &Value) {
  std::vector<uint64_t> Result;
  for (const Function &F : Value)
    for (const Instruction &I : instructions(F)) {
      if (auto *Load = dyn_cast<LoadInst>(&I))
        Result.push_back(Load->getAlign().value());
      else if (auto *Store = dyn_cast<StoreInst>(&I))
        Result.push_back(Store->getAlign().value());
    }
  return Result;
}

// Translation takes a missing load or store alignment from the module layout
// it starts with, so the admitted layout must be in place before translation.
void testLayoutBeforeTranslation() {
  constexpr StringLiteral Body = R"(
  llvm.func @fe2o3_mlir_layout(%p: !llvm.ptr<1>, %q: !llvm.ptr<1>) {
    %w = llvm.load %p : !llvm.ptr<1> -> i64
    %l = llvm.load %q : !llvm.ptr<1> -> !llvm.ptr<3>
    llvm.store %w, %p : i64, !llvm.ptr<1>
    llvm.store %l, %q : !llvm.ptr<3>, !llvm.ptr<1>
    llvm.return
  }
)";
  const std::string Text = mlirModule("", Body);
  const TargetStrings Strings = gfx942Strings();
  std::optional<std::pair<std::string, std::string>> Seen;
  LLVMContext AdmittedContext;
  auto Admitted = translateMlirLlvmTextInput(
      Text, "layout", AdmittedContext,
      [&](StringRef Triple, StringRef Layout) -> std::optional<std::string> {
        Seen.emplace(Triple.str(), Layout.str());
        return Strings.Layout;
      });
  if (!Admitted)
    fail(Twine("layout module was refused: ") + toString(Admitted.takeError()));
  require(Seen && Seen->first.empty() && Seen->second.empty(),
          "layout admission did not see the absent triple and layout");
  require((*Admitted)->getDataLayoutStr() == Strings.Layout,
          "translation did not keep the admitted layout");
  std::vector<uint64_t> Expected = {8, 4, 8, 4};
  require(memoryAlignments(**Admitted) == Expected,
          "translation did not take alignments from the admitted layout");

  LLVMContext DefaultContext;
  auto Default = translateMlirLlvmTextInput(Text, "layout", DefaultContext);
  if (!Default)
    fail(Twine("layout module without admission was refused: ") +
         toString(Default.takeError()));
  require(memoryAlignments(**Default) != Expected,
          "the default layout no longer changes these alignments, so the "
          "layout check proves nothing");
}

std::vector<uint8_t> testTextEquivalence() {
  std::vector<uint8_t> Reference;
  for (Header Variant : {Header::Machine, Header::Absent, Header::Legacy}) {
    std::vector<uint8_t> Text = requireSuccess(
        kernelRequest(InputKind::LlvmTextIr, textKernels(Variant)),
        "textual LLVM IR kernels");
    std::vector<uint8_t> Mlir = requireSuccess(
        kernelRequest(InputKind::MlirLlvmText, mlirKernels(Variant)),
        "MLIR kernels");
    requireSameBytes(Text, Mlir, "text and MLIR kernels");
    if (Reference.empty())
      Reference = Mlir;
    else
      requireSameBytes(Reference, Mlir, "MLIR header variants");
  }
  std::vector<uint8_t> Again = requireSuccess(
      kernelRequest(InputKind::MlirLlvmText, mlirKernels(Header::Machine)),
      "repeated MLIR kernels");
  requireSameBytes(Reference, Again, "repeated MLIR request");

  // Layout-derived alignments must agree for every header spelling.
  for (Header Variant : {Header::Machine, Header::Absent, Header::Legacy}) {
    auto LdsRequest = [&](InputKind Kind, const std::string &Text) {
      return makeV2Request(makeInput(Kind, bytesOf(Text)), LdsExports,
                           LdsFinals);
    };
    std::vector<uint8_t> Text = requireSuccess(
        LdsRequest(InputKind::LlvmTextIr, textModule(LdsText, Variant)),
        "textual LLVM IR LDS kernels");
    std::vector<uint8_t> Mlir = requireSuccess(
        LdsRequest(InputKind::MlirLlvmText,
                   mlirModule(mlirHeaderAttributes(Variant), LdsBody)),
        "MLIR LDS kernels");
    requireSameBytes(Text, Mlir, "text and MLIR LDS kernels");
  }

  for (OptimizationLevel Level :
       {OptimizationLevel::O0, OptimizationLevel::O1, OptimizationLevel::O2,
        OptimizationLevel::O3}) {
    for (bool Flags : {true, false}) {
      Request TextRequest =
          kernelRequest(InputKind::LlvmTextIr, textKernels(Header::Machine));
      Request MlirRequest =
          kernelRequest(InputKind::MlirLlvmText, mlirKernels(Header::Machine));
      TextRequest.LinkOptions = MlirRequest.LinkOptions = {Level, Flags, Flags};
      std::string Label = "O" + std::to_string(static_cast<unsigned>(Level)) +
                          (Flags ? " strip+verify" : " plain");
      requireSameBytes(requireSuccess(TextRequest, "textual " + Label),
                       requireSuccess(MlirRequest, "MLIR " + Label),
                       "text and MLIR kernels at " + Label);
    }
  }
  return Reference;
}

Request mlirRequest(StringRef Attributes, StringRef Body,
                    StringRef Target = "gfx942:xnack-") {
  return kernelRequest(InputKind::MlirLlvmText, mlirModule(Attributes, Body),
                       Target);
}

void testRefusals() {
  const std::string Machine = machineAttributes();
  const std::string Body = KernelsBody.str();

  requireRefusal(mlirRequest(Machine, "\n  %0 = arith.constant 1 : i32\n"),
                 Stage::InputValidation, "failed to parse or verify",
                 "unregistered arith dialect");
  requireRefusal(mlirRequest(Machine, "\n  func.func @f() { return }\n"),
                 Stage::InputValidation, "failed to parse or verify",
                 "unregistered func dialect");
  requireRefusal(mlirRequest(Machine, Body + "  module {}\n"),
                 Stage::InputValidation,
                 "uses operation 'builtin.module' outside the llvm and rocdl",
                 "nested builtin module");
  requireRefusal(
      mlirRequest(Machine,
                  Body + "  llvm.func @cast() {\n"
                         "    %0 = llvm.mlir.constant(1 : i32) : i32\n"
                         "    %1 = builtin.unrealized_conversion_cast %0 : "
                         "i32 to i64\n    llvm.return\n  }\n"),
      Stage::InputValidation,
      "uses operation 'builtin.unrealized_conversion_cast' outside",
      "unrealized conversion cast");

  // Translation would silently ignore a module-flag list spelled as a module
  // attribute instead of an `llvm.module_flags` operation.
  requireRefusal(
      mlirRequest(Machine + ", llvm.module_flags = [#llvm.mlir.module_flag<"
                            "error, \"amdgpu.xnack\", 1 : i32>]",
                  Body),
      Stage::InputValidation,
      "carries attribute 'llvm.module_flags' on 'builtin.module'",
      "module flags spelled as an attribute");
  // Upstream translation casts this attribute to a string unchecked.
  requireRefusal(mlirRequest("llvm.target_triple = 5 : i32", Body),
                 Stage::InputValidation,
                 "carries attribute 'llvm.target_triple' on 'builtin.module'",
                 "non-string target triple");
  requireRefusal(
      mlirRequest(Machine,
                  replaceAll(Body, "target_cpu = \"gfx942\",",
                             "target_cpu = \"gfx942\", rocdl.kernel,")),
      Stage::InputValidation, "carries attribute 'rocdl.kernel' on 'llvm.func'",
      "rocdl.kernel amendment");

  // Translation silently drops a parameter attribute it does not map.
  requireRefusal(
      mlirRequest(Machine, replaceAll(Body, "llvm.writeonly,", "llvm.noalis,")),
      Stage::InputValidation,
      "carries parameter attribute 'llvm.noalis' outside the admitted memory "
      "facts on 'llvm.func'",
      "misspelled parameter attribute");
  requireRefusal(mlirRequest(Machine, replaceAll(Body, "%alpha: f32 loc",
                                                 "%alpha: f32 {noalias} loc")),
                 Stage::InputValidation,
                 "arguments may only have dialect attributes",
                 "unprefixed parameter attribute");
  requireRefusal(
      mlirRequest(Machine,
                  replaceAll(Body, "llvm.writeonly,", "llvm.byval = f32,")),
      Stage::InputValidation,
      "carries parameter attribute 'llvm.byval' outside the admitted memory "
      "facts on 'llvm.func'",
      "ABI-changing parameter attribute");
  requireRefusal(
      mlirRequest(Machine, replaceAll(Body, "loc(\"n\")", "loc(\"data\")")),
      Stage::InputValidation,
      "names argument 2 of 'fe2o3_mlir_strided_scale' 'data', which is "
      "already used",
      "repeated argument name");

  requireRefusal(
      mlirRequest("llvm.data_layout = \"e-p:32:32\", llvm.target_triple = \"" +
                      gfx942Strings().Triple + "\"",
                  Body),
      Stage::InputValidation, "data layout does not match target machine",
      "foreign data layout");
  requireRefusal(
      mlirRequest("llvm.target_triple = \"x86_64-unknown-linux-gnu\"", Body),
      Stage::InputValidation, "target triple does not match AMDHSA",
      "foreign target triple");
  requireRefusal(
      mlirRequest(Machine,
                  replaceAll(Body, "500 : i32>]",
                             "500 : i32>, #llvm.mlir.module_flag<error, "
                             "\"amdgpu.xnack\", 1 : i32>]")),
      Stage::InputValidation, "amdgpu.xnack does not match request target",
      "conflicting xnack module flag");
  requireRefusal(
      mlirRequest(Machine, replaceAll(Body, "\"256,256\"", "\"128,128\"")),
      Stage::InputValidation, "verification failed",
      "flat workgroup bound below the required size");

  requireRefusal(mlirRequest(Machine, "\n  llvm.func @f( {\n"),
                 Stage::InputValidation, "failed to parse or verify",
                 "malformed MLIR");
  // Refusals echo input-chosen names; the stage bounds them itself and marks
  // the cut, both after the walk and on the parse path.
  std::string LongName(4 * MaxDiagnosticBytes, 'q');
  requireRefusal(
      mlirRequest(Machine, "\n  \"llvm." + LongName + "\"() : () -> ()\n"),
      Stage::InputValidation, "uses unregistered operation 'llvm.qqq",
      "oversized walk refusal", "<truncated>");
  requireRefusal(mlirRequest(Machine, "\n  llvm." + LongName + "\n"),
                 Stage::InputValidation, "failed to parse or verify",
                 "oversized parse diagnostic", "<truncated>");
  // Errors are reported before warnings, so warnings cannot crowd them out.
  std::string Warned;
  for (int I = 0; I < 8; ++I)
    Warned += "{-# external_resources: { qqq" + std::to_string(I) +
              std::string(300, 'r') + ": { k: \"v\" } } #-}\n";
  Warned += "module {\n  llvm.qqqerror\n}\n";
  requireRefusal(
      kernelRequest(InputKind::MlirLlvmText, Warned), Stage::InputValidation,
      "failed to parse or verify: error: ", "error after many warnings",
      "custom op 'llvm.qqqerror' is unknown");
  for (StringRef Truncated : {"module", "module attributes {a = \"x"})
    requireRefusal(kernelRequest(InputKind::MlirLlvmText, Truncated),
                   Stage::InputValidation, "failed to parse or verify",
                   "truncated MLIR payload");
  requireRefusal(mlirRequest(Machine, Body + "  \"llvm.qqq\"() : () -> ()\n"),
                 Stage::InputValidation,
                 "uses unregistered operation 'llvm.qqq'",
                 "unregistered llvm operation");

  // Translation passes these values to LLVM unchecked, where they assert or
  // build invalid IR.
  const std::string Rangeless = "%n: i32 loc";
  for (auto [Attribute, Expected] :
       {std::pair{"{llvm.range = 5 : i32}",
                  "carries parameter attribute 'llvm.range' without a "
                  "constant range value"},
        {"{llvm.range}", "carries parameter attribute 'llvm.range' without a "
                         "constant range value"},
        {"{llvm.range = #llvm.constant_range<i32, 5, 5>}",
         "carries the constant range <i32, 5, 5> with equal bounds"},
        {"{llvm.range = #llvm.constant_range<i64, 0, 5>}",
         "carries parameter attribute 'llvm.range' whose width does not "
         "match its integer value"}})
    requireRefusal(
        mlirRequest(Machine,
                    replaceAll(Body, Rangeless,
                               "%n: i32 " + std::string(Attribute) + " loc")),
        Stage::InputValidation, Expected, "parameter range",
        "on 'llvm.func', which this worker does not admit");
  const std::string Caller = R"(
  llvm.func @fe2o3_mlir_sink(i32)
  llvm.func @fe2o3_mlir_caller(%x: i32) {
    llvm.call @fe2o3_mlir_sink(%x) : (i32 @ATTR@) -> ()
    llvm.return
  }
)";
  for (auto [Attribute, Expected] :
       {std::pair{"{llvm.noalias = 5 : i64}",
                  "carries parameter attribute 'llvm.noalias' with a value"},
        {"{llvm.range = #llvm.constant_range<i32, 5, 5>}",
         "carries the constant range <i32, 5, 5> with equal bounds"}})
    requireRefusal(
        mlirRequest(Machine, Body + replaceAll(Caller, "@ATTR@", Attribute)),
        Stage::InputValidation, Expected, "call-site parameter attribute",
        "on 'llvm.call'");
  requireRefusal(
      mlirRequest(Machine, replaceAll(Body, "%tid = rocdl.workitem.id.x : i32",
                                      "%tid = rocdl.workitem.id.x range "
                                      "<i32, 7, 7> : i32")),
      Stage::InputValidation,
      "carries the constant range <i32, 7, 7> with equal bounds on "
      "'rocdl.workitem.id.x'",
      "equal-bound work-item range");

  const std::string Load = "%va = llvm.load %pa <alignment = 4>";
  for (StringRef Alignment : {"3", "8589934592"})
    requireRefusal(
        mlirRequest(Machine, replaceAll(Body, Load,
                                        "%va = llvm.load %pa <alignment = " +
                                            Alignment.str() + ">")),
        Stage::InputValidation,
        "carries alignment " + Alignment.str() +
            ", which is not a power of two within LLVM's bound on "
            "'llvm.load'",
        "load alignment");
  for (StringRef Alignment : {"3", "0"})
    requireRefusal(
        mlirRequest(Machine, replaceAll(Body,
                                        "%data: !llvm.ptr<1> {llvm.noalias, "
                                        "llvm.noundef, llvm.align = 4 : i64}",
                                        "%data: !llvm.ptr<1> {llvm.noalias, "
                                        "llvm.noundef, llvm.align = " +
                                            Alignment.str() + " : i64}")),
        Stage::InputValidation,
        "carries parameter attribute 'llvm.align' with a value that is not a "
        "power of two within LLVM's bound",
        "parameter alignment");
  requireRefusal(mlirRequest(Machine, Body + R"(
  llvm.func @fe2o3_mlir_alloca() {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %0 = llvm.alloca %c1 x i32 {alignment = 3 : i64} : (i32) -> !llvm.ptr<5>
    llvm.return
  }
)"),
                 Stage::InputValidation,
                 "carries alignment 3, which is not a power of two within "
                 "LLVM's bound on 'llvm.alloca'",
                 "alloca alignment");
  requireRefusal(
      mlirRequest(Machine, replaceAll(Body, "attributes {reqd_work_group_size",
                                      "attributes {alignment = 3 : i64, "
                                      "reqd_work_group_size")),
      Stage::InputValidation,
      "carries alignment 3, which is not a power of two within LLVM's bound "
      "on 'llvm.func'",
      "function alignment");

  requireRefusal(mlirRequest(Machine, Body + R"(
  llvm.func @fe2o3_mlir_asm(%p: !llvm.ptr) {
    llvm.inline_asm operand_attrs = [{elementtype = i32}] "", "=*m" %p : (!llvm.ptr) -> ()
    llvm.return
  }
)"),
                 Stage::InputValidation,
                 "carries inline-assembly operand attributes on "
                 "'llvm.inline_asm'",
                 "inline-assembly operand attributes");
  requireRefusal(mlirRequest(Machine, Body + R"(
  llvm.func @fe2o3_mlir_asm() {
    llvm.inline_asm "s_nop 0", "v" : () -> ()
    llvm.return
  }
)"),
                 Stage::InputValidation,
                 "carries inline-assembly constraints that do not match its "
                 "operands",
                 "inline-assembly constraints");

  for (StringRef Name : {"llvm.dbg.cu", "llvm.module.flags"})
    requireRefusal(
        mlirRequest(Machine, Body + "  llvm.named_metadata \"" + Name.str() +
                                 "\" [#llvm.md_node<#llvm.md_const<1 : "
                                 "i32>>]\n"),
        Stage::InputValidation,
        "uses operation 'llvm.named_metadata', which this worker does not "
        "admit",
        "named metadata");

  const std::string Alpha = "loc(\"alpha\")";
  for (auto [Location, Expected] :
       {std::pair<std::string, std::string>{"loc(\"al\\00pha\")",
                                            "with a NUL byte in its name"},
        {"loc(\"" + std::string(1025, 'a') + "\")",
         "with a name longer than LLVM's 1024-byte local name limit"},
        {"loc(fused[\"alpha\", \"beta\"])",
         "through a name nested inside another location"},
        {"loc(\"alpha\"(\"beta\"))", "with nested name locations"}})
    requireRefusal(mlirRequest(Machine, replaceAll(Body, Alpha, Location)),
                   Stage::InputValidation,
                   "names argument 1 of 'fe2o3_mlir_strided_scale' " + Expected,
                   "argument name");
  requireRefusal(mlirRequest(Machine, replaceAll(Body, "@fe2o3_mlir_vecadd",
                                                 "@\"fe2o3\\00nul\"")),
                 Stage::InputValidation,
                 "names a symbol with a NUL byte on 'llvm.func'",
                 "NUL symbol name");

  std::string NonAscii = mlirKernels(Header::Machine);
  NonAscii.insert(NonAscii.find("llvm.module_flags"), "// \xc3\xa9\n");
  requireRefusal(kernelRequest(InputKind::MlirLlvmText, NonAscii),
                 Stage::InputValidation, "noncanonical textual MLIR bytes",
                 "non-ASCII MLIR bytes");
}

// src/main.cpp's authenticated machine-effect mode applies these limits, which
// deny thread and process creation and bound address space, data and files;
// the plain link mode applies none. The MLIR stage must complete under them,
// so a child process forked before any LLVM or MLIR state exists applies them,
// executes the request and hands its HSACO back for comparison.
int testWithoutThreadCreation() {
  int Pipe[2];
  require(pipe(Pipe) == 0, "pipe failed");
  pid_t Child = fork();
  require(Child >= 0, "fork failed");
  if (Child == 0) {
    close(Pipe[0]);
    struct rlimit Zero = {0, 0};
    struct rlimit AddressSpace = {4ULL << 30, 4ULL << 30};
    struct rlimit Data = {2ULL << 30, 2ULL << 30};
    struct rlimit Files = {16ULL << 20, 16ULL << 20};
    if (setrlimit(RLIMIT_AS, &AddressSpace) != 0 ||
        setrlimit(RLIMIT_DATA, &Data) != 0 ||
        setrlimit(RLIMIT_FSIZE, &Files) != 0 ||
        setrlimit(RLIMIT_CORE, &Zero) != 0 ||
        setrlimit(RLIMIT_NPROC, &Zero) != 0)
      _exit(10);
    pthread_t Thread;
    int Created = pthread_create(
        &Thread, nullptr, [](void *) -> void * { return nullptr; }, nullptr);
    if (Created == 0) {
      pthread_join(Thread, nullptr);
      _exit(11); // RLIMIT_NPROC is not enforced here (for example, as root).
    }
    Response Result = execute(
        kernelRequest(InputKind::MlirLlvmText, mlirKernels(Header::Machine)));
    if (!Result.LinkedOutput) {
      printDiagnostics(Result);
      _exit(12);
    }
    ArrayRef<uint8_t> Bytes = Result.LinkedOutput->Bytes;
    while (!Bytes.empty()) {
      ssize_t Written = write(Pipe[1], Bytes.data(), Bytes.size());
      if (Written < 0 && errno == EINTR)
        continue;
      if (Written <= 0)
        _exit(13);
      Bytes = Bytes.drop_front(static_cast<size_t>(Written));
    }
    _exit(0);
  }
  close(Pipe[1]);
  std::vector<uint8_t> ChildBytes;
  uint8_t Buffer[65536];
  for (;;) {
    ssize_t Read = read(Pipe[0], Buffer, sizeof(Buffer));
    if (Read < 0 && errno == EINTR)
      continue;
    require(Read >= 0, "reading the thread-free child's HSACO failed");
    if (Read == 0)
      break;
    ChildBytes.insert(ChildBytes.end(), Buffer, Buffer + Read);
  }
  close(Pipe[0]);
  int Status = 0;
  require(waitpid(Child, &Status, 0) == Child, "waitpid failed");
  require(WIFEXITED(Status), "thread-free MLIR child did not exit normally");
  int Code = WEXITSTATUS(Status);
  if (Code == 11) {
    errs() << "note: RLIMIT_NPROC does not deny threads here; "
              "thread-free check skipped\n";
    return 77;
  }
  require(Code == 0,
          Twine("thread-free MLIR child failed with code ") + Twine(Code));
  std::vector<uint8_t> Expected = requireSuccess(
      kernelRequest(InputKind::LlvmTextIr, textKernels(Header::Machine)),
      "textual LLVM IR kernels");
  requireSameBytes(Expected, ChildBytes, "thread-free MLIR kernels");
  outs() << "thread-free MLIR input check passed (" << ChildBytes.size()
         << "-byte HSACO)\n";
  return 0;
}

std::string readCorpusFile(const Twine &Path) {
  auto Buffer = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                      /*RequiresNullTerminator=*/false);
  if (!Buffer)
    fail(Twine("cannot read ") + Path + ": " + Buffer.getError().message());
  return (*Buffer)->getBuffer().str();
}

struct CorpusSymbols {
  std::vector<std::string> Exports;
  std::vector<std::string> Finals;
  unsigned Kernels = 0;
  // The worker's required imports. The corpus mode stages no providers, so a
  // pair with any is not linked.
  std::vector<std::string> Imports;
  // The imports that only trivially dead instructions use.
  std::vector<std::string> DeadImports;

  bool sameDefinitions(const CorpusSymbols &Other) const {
    return Exports == Other.Exports && Finals == Other.Finals &&
           Kernels == Other.Kernels;
  }
};

bool onlyDeadInstructionsUse(const GlobalValue &Value) {
  return llvm::all_of(Value.users(), [](const User *Using) {
    const auto *UsingInstruction = dyn_cast<Instruction>(Using);
    return UsingInstruction && UsingInstruction->use_empty() &&
           !UsingInstruction->mayHaveSideEffects() &&
           !UsingInstruction->isTerminator();
  });
}

std::unique_ptr<Module> parseCorpusText(StringRef Text,
                                        const std::string &Label,
                                        LLVMContext &Context) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<Module> Parsed =
      parseAssemblyString(Text, Diagnostic, Context);
  if (!Parsed) {
    std::string Message;
    raw_string_ostream Stream(Message);
    Diagnostic.print(Label.c_str(), Stream);
    fail(Twine(Label) + ": textual LLVM IR does not parse: " + Stream.str());
  }
  return Parsed;
}

// The module names its closure: every external definition is exported and
// every kernel adds its descriptor. Imports need providers the corpus mode
// does not stage, so such pairs are linked by neither path.
CorpusSymbols corpusSymbols(const Module &Parsed) {
  // Mirrors the worker's symbol inspection: a used non-intrinsic declaration
  // is an import, and every other external value is a public definition.
  CorpusSymbols Result;
  for (const GlobalValue &Value : Parsed.global_values()) {
    if (!Value.hasName() || Value.hasLocalLinkage() ||
        Value.hasAppendingLinkage())
      continue;
    std::string Name = Value.getName().str();
    const auto *FunctionValue = dyn_cast<Function>(&Value);
    if (Value.isDeclarationForLinker()) {
      if (!Value.use_empty() &&
          (!FunctionValue || !FunctionValue->isIntrinsic())) {
        Result.Imports.push_back(Name);
        if (onlyDeadInstructionsUse(Value))
          Result.DeadImports.push_back(Name);
      }
      continue;
    }
    Result.Exports.push_back(Name);
    Result.Finals.push_back(Name);
    if (FunctionValue &&
        FunctionValue->getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      Result.Finals.push_back(Name + ".kd");
      ++Result.Kernels;
    }
  }
  llvm::sort(Result.Exports);
  llvm::sort(Result.Finals);
  llvm::sort(Result.Imports);
  llvm::sort(Result.DeadImports);
  return Result;
}

std::string describe(const CorpusSymbols &Symbols) {
  std::string Result = "kernels=" + std::to_string(Symbols.Kernels);
  for (const std::string &Name : Symbols.Finals)
    Result += " " + Name;
  if (!Symbols.Imports.empty())
    Result += " imports " + llvm::join(Symbols.Imports, ", ");
  return Result;
}

// Runs for every pair, including those neither path can link here: the
// worker's MLIR stage must translate the module and yield the textual
// module's closure, and every definition must keep its linkage, calling
// convention, and argument names (restored from `loc("...")`). The imports
// must agree except for textual ones only dead instructions use:
// mlir-translate's IRBuilder folds the all-zero-index getelementptr of a
// dynamic LDS global, so the MLIR module keeps no use of it. Returns those.
std::vector<std::string> requireSameClosure(const Module &Text, StringRef Mlir,
                                            StringRef Stem) {
  LLVMContext Context;
  auto Translated = translateMlirLlvmTextInput(Mlir, Stem, Context);
  if (!Translated)
    fail(Twine(Stem) +
         ": MLIR translation failed: " + toString(Translated.takeError()));
  const Module &Value = **Translated;
  CorpusSymbols Expected = corpusSymbols(Text);
  CorpusSymbols Actual = corpusSymbols(Value);
  std::vector<std::string> LiveImports;
  std::set_difference(Expected.Imports.begin(), Expected.Imports.end(),
                      Expected.DeadImports.begin(), Expected.DeadImports.end(),
                      std::back_inserter(LiveImports));
  std::vector<std::string> MissingImports;
  std::set_difference(Expected.Imports.begin(), Expected.Imports.end(),
                      Actual.Imports.begin(), Actual.Imports.end(),
                      std::back_inserter(MissingImports));
  require(Expected.sameDefinitions(Actual) &&
              std::includes(Expected.Imports.begin(), Expected.Imports.end(),
                            Actual.Imports.begin(), Actual.Imports.end()) &&
              std::includes(Actual.Imports.begin(), Actual.Imports.end(),
                            LiveImports.begin(), LiveImports.end()),
          Twine(Stem) + ": MLIR closure " + describe(Actual) +
              " differs from textual " + describe(Expected));
  unsigned Definitions = 0;
  for (const Function &Defined : Text) {
    if (Defined.isDeclaration())
      continue;
    ++Definitions;
    std::string Label = (Twine(Stem) + ": " + Defined.getName()).str();
    const Function *Other = Value.getFunction(Defined.getName());
    require(Other && !Other->isDeclaration(), Twine(Label) + " is not defined");
    require(Other->getLinkage() == Defined.getLinkage(),
            Twine(Label) + " changed linkage");
    require(Other->getCallingConv() == Defined.getCallingConv(),
            Twine(Label) + " changed calling convention");
    require(Other->arg_size() == Defined.arg_size(),
            Twine(Label) + " changed its parameter count");
    for (unsigned Index = 0; Index < Defined.arg_size(); ++Index)
      require(Other->getArg(Index)->getName() ==
                  Defined.getArg(Index)->getName(),
              Twine(Label) + " renamed parameter " + Twine(Index) + " from '" +
                  Defined.getArg(Index)->getName() + "' to '" +
                  Other->getArg(Index)->getName() + "'");
  }
  unsigned OtherDefinitions = llvm::count_if(
      Value, [](const Function &Defined) { return !Defined.isDeclaration(); });
  require(OtherDefinitions == Definitions,
          Twine(Stem) + ": MLIR defines " + Twine(OtherDefinitions) +
              " functions, the textual module " + Twine(Definitions));
  return MissingImports;
}

// `--corpus DIR`: DIR holds `{stem}.{gfx942|gfx950}.ll` and `.mlir` pairs
// written by the Rust differential tests under FE2O3_MLIR_DUMP_DIR. Every
// pair must translate to the textual module's closure (see
// requireSameClosure); each pair without imports must also link to
// byte-identical HSACO through the full V2 pipeline.
int testCorpus(StringRef Directory) {
  require(mlirLlvmTextInputSupported(),
          "--corpus needs a worker built with MLIR input");
  std::set<std::string> TextStems;
  std::set<std::string> MlirStems;
  std::error_code ErrorCode;
  for (sys::fs::directory_iterator Entry(Directory, ErrorCode), End;
       !ErrorCode && Entry != End; Entry.increment(ErrorCode)) {
    StringRef Extension = sys::path::extension(Entry->path());
    std::string Stem = sys::path::stem(Entry->path()).str();
    if (Extension == ".ll")
      TextStems.insert(Stem);
    else if (Extension == ".mlir")
      MlirStems.insert(Stem);
  }
  require(!ErrorCode,
          Twine("cannot list ") + Directory + ": " + ErrorCode.message());
  require(!TextStems.empty(), Twine(Directory) + " holds no .ll files");
  for (const std::string &Stem : MlirStems)
    require(TextStems.count(Stem),
            Twine(Stem) + ".mlir has no textual .ll partner");
  for (const std::string &Stem : TextStems)
    require(MlirStems.count(Stem), Twine(Stem) + ".ll has no .mlir partner");
  unsigned Compared = 0;
  unsigned Skipped = 0;
  unsigned DeadUseOnly = 0;
  unsigned MultiKernel = 0;
  for (const std::string &Stem : TextStems) {
    StringRef Cpu = sys::path::extension(Stem);
    require(Cpu == ".gfx942" || Cpu == ".gfx950",
            Twine(Stem) + ": the stem does not end in .gfx942 or .gfx950");
    std::string Target = Cpu.drop_front().str() + ":xnack-";
    SmallString<256> Base(Directory);
    sys::path::append(Base, Stem);
    std::string Text = readCorpusFile(Base + ".ll");
    std::string Mlir = readCorpusFile(Base + ".mlir");
    LLVMContext Context;
    std::unique_ptr<Module> Parsed = parseCorpusText(Text, Stem, Context);
    std::vector<std::string> DeadOnly = requireSameClosure(*Parsed, Mlir, Stem);
    CorpusSymbols Symbols = corpusSymbols(*Parsed);
    if (!Symbols.Imports.empty()) {
      outs() << "corpus " << Stem << ": closure identical";
      if (!DeadOnly.empty()) {
        outs() << " but for the textual import of "
               << llvm::join(DeadOnly, ", ") << " by dead instructions only";
        ++DeadUseOnly;
      }
      ArrayRef<std::string> Shown(Symbols.Imports);
      outs() << ", link skipped (imports "
             << llvm::join(Shown.take_front(3), ", ");
      if (Shown.size() > 3)
        outs() << " and " << Shown.size() - 3 << " more";
      outs() << ")\n";
      outs().flush();
      ++Skipped;
      continue;
    }
    require(Symbols.Kernels > 0, Twine(Stem) + " defines no kernel");
    auto RequestFor = [&](InputKind Kind, const std::string &Source) {
      return makeV2Request(makeInput(Kind, bytesOf(Source)), Symbols.Exports,
                           Symbols.Finals, Target);
    };
    std::vector<uint8_t> TextBytes =
        requireSuccess(RequestFor(InputKind::LlvmTextIr, Text),
                       Stem + " (textual LLVM IR)", Symbols.Kernels);
    std::vector<uint8_t> MlirBytes =
        requireSuccess(RequestFor(InputKind::MlirLlvmText, Mlir),
                       Stem + " (MLIR)", Symbols.Kernels);
    requireSameBytes(TextBytes, MlirBytes, Stem);
    outs() << "corpus " << Stem << ": " << TextBytes.size()
           << "-byte HSACO identical (" << Symbols.Kernels << " kernel"
           << (Symbols.Kernels == 1 ? "" : "s") << ")\n";
    outs().flush();
    ++Compared;
    if (Symbols.Kernels > 1)
      ++MultiKernel;
  }
  require(Compared > 0, "no corpus pair was compared");
  outs() << "MLIR corpus passed: " << Compared << " identical (" << MultiKernel
         << " multi-kernel), " << Skipped << " closure-only (" << DeadUseOnly
         << " with dead-use textual imports)\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  require(mlirLlvmTextInputSupported() == (FE2O3_WORKER_MLIR_INPUT != 0),
          "MLIR support flag disagrees with the build definition");
  if (argc == 2 && StringRef(argv[1]) == "--thread-free") {
    // Its own ctest entry: the child must start before this process creates
    // any LLVM or MLIR state. 77 is the registered skip code.
    if (!mlirLlvmTextInputSupported())
      return 77;
    return testWithoutThreadCreation();
  }
  if (argc == 3 && StringRef(argv[1]) == "--corpus")
    return testCorpus(argv[2]);
  require(argc == 1, "usage: fe2o3-worker-mlir-input-tests "
                     "[--thread-free | --corpus DIR]");
  testDecodeGate();
  if (!mlirLlvmTextInputSupported()) {
    testRefusedWithoutSupport();
    outs() << "MLIR input refused by this worker build\n";
    return 0;
  }
  testDirectTranslation();
  testAdmittedForms();
  testLayoutBeforeTranslation();
  std::vector<uint8_t> Expected = testTextEquivalence();
  testRefusals();
  outs() << "MLIR input tests passed (" << Expected.size()
         << "-byte HSACO identical to textual LLVM IR)\n";
  return 0;
}
