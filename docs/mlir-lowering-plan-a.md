# План A: MLIR-лоуэринг внутри измеряемого воркера + переход на LLVM 24

Статус: фазы 0-2 выполнены для инженерного пути, production не затронут; ничего не закоммичено.
Ветка: `mlir-lowering-llvm24`. Целевой toolchain: LLVM `24.0.0git`
(`cea26e4798f7899d9c40259aad6f793038a1f01f`), сборка `/home/mrsidims/LLVM-work/build`
(`LLVM_LINK_LLVM_DYLIB=ON`, assertions ON, RTTI/EH OFF, проекты clang+lld+mlir).

## 0. Рамки и решения

| Решение | Выбор | Следствие |
|---|---|---|
| Где живёт MLIR | **Внутри C++ воркера** (вариант A) | В Rust по-прежнему нет `llvm-sys`/`mlir-sys`; LLVM и MLIR остаются только в измеряемом процессе |
| Куда ставим | **Сначала engineering-путь, за флагом** | Production получает только перепин версии LLVM; `llvm_ir_sha256`, `PRODUCTION_*` layout-константы, lineage и квитанции не меняются |
| OCML | **Пока нет** | В скоупе только math-free ядра; проверка через HSACO, KIR-симулятор и сравнение ISA; прогоны на GPU делает пользователь |
| Уровень MLIR | `llvm` + `rocdl` диалекты (не `gpu`/`arith`) | Эмиттер детерминированно печатает текст в диалекте `llvm`; никаких конверсионных пассов (`convert-gpu-to-rocdl` подставляет устаревший datalayout) |

Криптография/квитанции нужны на уровне рантайма, а не внутри компилятора, поэтому MLIR-путь не
обязан воспроизводить байт-в-байт текущий LLVM IR: он обязан давать **семантически
эквивалентный** модуль, а на G1-подмножестве и **идентичный машинный код**.

## 1. Фаза 0: миграция на LLVM 24 (выполнено, не закоммичено)

### 1.1 Что изменилось в LLVM 24 для AMDGPU (проверено по исходникам и эмпирически)

| Изменение | Где проявилось | Что сделано |
|---|---|---|
| Triple с subarch: `amdgpu9.42-amd-amdhsa`, `amdgpu9.50-amd-amdhsa`; `lookupTarget(ArchName, Triple&)` перезаписывает triple на `amdgpu-amd-amdhsa` | воркер, gfx950 one-stop fixture | `amdgpuTriple(Cpu)` + `lookupTarget(const Triple&, Err)`; контракт модуля принимает legacy `amdgcn-amd-amdhsa` и машинный triple |
| `xnack`/`sramecc` больше не subtarget features: только module flags `amdgpu.xnack`/`amdgpu.sramecc` (0 = off, ≠0 = on, нет флага = Any). В feature-строке TargetMachine → `DS_Error` в AsmPrinter | воркер, MC в тестах | Пустые TM features; воркер добавляет module flag `amdgpu.xnack = 0` для `:xnack-` |
| `amdhsa.target` = `amdgpu9.42-amd-amdhsa-unknown-gfx942:xnack-` (e_flags не изменились) | `fe2o3-hsaco` метаданные, воркер, золотые файлы | Принимаются обе формы target-ID со строгой проверкой subarch |
| Datalayout получил `p10`-`p15`; LLParser **не** апгрейдит layout текстового IR (BitcodeReader апгрейдит) | воркер | `AdmitLayout` применяет `UpgradeDataLayoutString` для текста |
| `"denormal-fp-math-f32"` → `denormal_fpenv` только через `UpgradeFunctionAttributes` (вызывается из BitcodeReader) | воркер | `UpgradeFunctionAttributes` для текстовых модулей |
| Новое правило Verifier (`VerifierAMDGPU.cpp:75-131`): при `reqd_work_group_size` атрибут `amdgpu-flat-work-group-size` обязан быть `"P,P"`, P = произведение | `lowering.rs` с диапазонными launch policy | Диапазон сужается до точного размера; exact-политики байт-идентичны (дайджесты не меняются) |
| C++ API: `MCContext(Triple&, AsmInfo&, RegInfo&, STI&)`, `StringSwitch::Cases({..})`, `PassBuilder(Ctx,..)`, `getMCSubtargetInfo()` возвращает ссылку, `SubtargetFeatureKV::key()`, `BranchInst` → `UncondBrInst`/`CondBrInst`, `setModuleInlineAsm(GlobalAsmFragment)`, `FileOutputBuffer::F_mmap` → `F_in_memory` | воркер, тесты воркера | Портировано |
| `llvm-readobj` печатает `Arch: amdgpu` вместо `amdgcn` | `fe2o3-amdgcn-model/tests/lowering.rs` | Принимаются обе формы |

### 1.2 Результаты

- Воркер под LLVM 24 собирается (RUNPATH на `build/lib`, `NEEDED libLLVM.so.24.0git`, работает
  под `env -i`); ctest 7/7.
- `fe2o3-hsaco`: все тесты зелёные.
- `fe2o3-amdgcn-model`: 566/566 обычных тестов зелёные с инструментами LLVM 24.
- ROCm-гейтнутые (`#[ignore]`) тесты `fe2o3-amdgcn-model` под LLVM 24 через mount namespace
  (`unshare -rm`, подмена `/opt/rocm/llvm/bin`): 18/21. Падают три теста, все на **прямом
  clang-пути** (минуя воркер):
  1. два теста: `Can't create a MachineFunction using a Module with a Target-incompatible
     DataLayout`. Причина: явный legacy-layout без `p10`-`p15`, который LLParser не
     апгрейдит. Через воркер всё работает;
  2. `rocm_compiles_the_golden_to_an_amdgpu_code_object`: ожидает `EF_AMDGPU_FEATURE_XNACK_OFF_V4`,
     а получает XNACK_ANY. Это **настоящая семантическая регрессия прямого пути** (см. 2.1),
     тест оставлен падать намеренно.
- Фикстуры воркера: 8/9 собираются и проходят свои self-contained проверки.
  `gfx950-one-stop-fixture` собирается, на O3 инструкции совпадают с эталоном, но падает на
  проверке prefetch-padding (см. 2.3). Эталон оставлен без изменений как fail-closed
  свидетельство toolchain.

### 1.3 Что осталось в фазе 0

- `tools/fe2o3-host-lld` (полностью статический, `--wrap`) и
  `tools/fe2o3-instruction-native-observation` (`FATAL_ERROR`, если есть `TARGET LLVM`) не
  собираются против dylib-LLVM. Им нужна отдельная статическая сборка LLVM 24
  (`LLVM_LINK_LLVM_DYLIB=OFF`, `BUILD_SHARED_LIBS=OFF`, только `AMDGPU;X86`, проекты `lld`).
- Не перепинены: upstream 22.1.8 (`checked_binary.rs`, host-lld) и ROCm 22.0.0git
  (gfx950 rocm-7.2.x профили). Это правильно: профили ROCm привязаны к манифесту и
  выбираются явно. Перепин production-констант будет только после фазы 4.
- Исторические скрипты свидетельств (`scripts/s09-debug-check.py`, `scripts/gfx942_compiler_evidence.py`)
  не трогаем: это записи о прошлом toolchain.
- `tools/fe2o3-instruction-native-observation/WorkerSourcePins.cmake` пинит 13 исходников
  воркера по SHA-256 к базе `harsh-nod/fe2o3@c60cd746`. Пин `CMakeLists.txt` не совпадал ещё
  **до** этой ветки (HEAD: 19859 байт, пин: 18827). Ветка меняет его и ещё 5 запиненных файлов
  (`WorkerProtocol.h`, `WorkerProtocol.cpp`, `WorkerPipeline.cpp`, `WorkerMachineEffect.cpp`,
  `tests/OrderedProgramWorkerSupport.inc`). Перепин делается после ревью, вместе со
  статической сборкой LLVM 24 для этого инструмента.
- Тесты `cargo-fe2o3` в этом окружении (WSL2) частично падают по причинам, не связанным с
  миграцией:
  - нет reviewed native Python (`FE2O3_TEST_NATIVE_PYTHON`, нужен python3.12/3.13);
  - `tools/fe2o3-static-preexec-launcher` не компилируется: в заголовках glibc нет `F_SEAL_EXEC`;
  - `profile_command`/`profile_cli` тесты зависят от capture-воркера и KFD хоста.

## 2. Ловушки LLVM 24, которые обязан учитывать MLIR-путь

### 2.1 xnack молча превращается в Any

`clang -x ir -mcpu=gfx942:xnack-` на LLVM 24 выдаёт `XNACK_ANY` (e_flags `0x54C`): функциональный
`-xnack` игнорируется, а module flag никто не добавляет. С `!{i32 1, !"amdgpu.xnack", i32 0}`
получается `0x64C`.

**Правила:**
1. Текстовый IR из `lowering.rs` исполняем только через воркер (воркер добавляет флаг).
2. MLIR-эмиттер module flags **не пишет вообще** (ни `amdgpu.xnack`, ни code object version;
   это проверяет тест). Флаги, как и для текстового IR, ставит воркер в
   `setAndCheckModuleContract`. Если флаг во входе уже есть и не совпадает с запрошенным
   target-ID, это отказ, а не перезапись. Воркер принимает и `llvm.module_flags` во входе (тест
   на конфликт есть), но эмиттер этим не пользуется: один владелец флага, одно место проверки.
3. В инспекции HSACO e_flags XNACK сверяются с target-ID запроса (уже делается в `fe2o3-hsaco`).
4. `-xnack` в `target-features` функции на LLVM 24 вызывает предупреждение
   `'-xnack' is not a recognized feature for this target (ignoring feature)`. Оно появляется
   на обоих путях одинаково и ни на что не влияет. `llc -mattr=-xnack` LLVM 24 отвергает,
   поэтому дифференциальный тест вызывает `llc` без `-mattr`.

### 2.2 O0 + inline asm на всё тело читает необъявленные ABI-регистры

На O0 LLVM 24 перед whole-body inline asm вставляет загрузку неиспользуемого kernarg:
`s_load_dwordx2 s[0:1], s[0:1], 0x0`. Если asm читает `s[0:1]` без объявления в constraints, он
видит гонку или затёртый указатель kernarg. На O3 код совпадает с эталоном.

**Правило:** naked/whole-body asm-пути либо объявляют все читаемые ABI-регистры во входных
constraints, либо не допускаются на O0. MLIR-эмиттер генерирует ровно один asm, как и
`lowering.rs`: физический барьер `llvm.inline_asm has_side_effects "s_barrier", ""`. Он не
читает ABI-регистров, поэтому ловушка его не касается. Любой другой inline asm эмиттер
отвергает.

### 2.3 gfx950 prefetch padding

Upstream LLVM 24 не вставляет 1068-байтовый `S_NOP` prefetch-padding (.text 92 байта на O0 и
84 на O3 против требуемых 1152). Похоже, это специфика ROCm-LLVM. Для gfx950 на upstream
LLVM нужен новый проверенный профиль; до тех пор gfx950 в MLIR-пути только compile-only.

### 2.4 Прочее

- `convert-gpu-to-rocdl` подставляет устаревший `amdgcnDataLayout`
  (`LowerGpuOpsToROCDLOps.cpp:172-177, 742-748`), поэтому его не используем.
- `rocdl.kernel` добавляет `uniform-work-group-size` и flat size по умолчанию `1,256`
  (`ROCDLToLLVMIRTranslation.cpp:54-80`), поэтому его не используем.
- `rocdl.max_flat_work_group_size` печатает `"1,N"`, а это нарушает правило Verifier из 1.1.
  Нужен `passthrough`.
- Без OCKL нельзя опускать `gpu.block_dim`/`gpu.grid_dim` через `__ockl_*`: размеры берутся из
  `llvm.amdgcn.dispatch.ptr` напрямую, как в `lowering.rs`.

## 3. Архитектура

```
#[kernel(typed)] ─▶ rustc-codegen-fe2o3 ─▶ MIR ─▶ KIR (verify_module)
                                                    │
                     ┌──────────────────────────────┴───────────────────────────┐
                     ▼ (production, без изменений)                              ▼ (engineering, флаг)
           lowering.rs: текстовый LLVM IR                      lowering/mlir_text_v1.rs: MLIR text
           InputKind::LlvmTextIr = 3                           (диалекты llvm + rocdl)
                     │                                         InputKind::MlirLlvmText = 4
                     ▼                                                    ▼
   ┌─────────────────────────── fe2o3-llvm-link-worker (измеряемый процесс) ─────────────────┐
   │ decode ─▶ [MLIR: parse (Threading::DISABLED) ─▶ allowlist диалектов ─▶                   │
   │            translateModuleToLLVMIR] ─▶ layout admission ─▶ module contract ─▶            │
   │ (OCML link) ─▶ opt ─▶ codegen ─▶ in-process LLD ─▶ инспекция HSACO ─▶ ответ              │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
```

Главный принцип: MLIR-стадия только **порождает `llvm::Module`**. Дальше модуль идёт тем же
кодом, что и текстовый IR: тот же layout admission, тот же `setAndCheckModuleContract`, тот же
`verifyModule`, та же оптимизация, codegen, LLD и инспекция. Новых путей к машинному коду нет.

### 3.1 Гейтинг

- CMake-опция `FE2O3_WORKER_ENABLE_MLIR_INPUT` (по умолчанию `OFF`), требует `MLIR_DIR`.
- Значение опции и версия MLIR входят в `FE2O3_WORKER_MEASUREMENT_PREIMAGE`. Поэтому воркер с
  MLIR всегда имеет **другой** build identity и не может выдать себя за production-воркер,
  запиненный по дайджесту.
- Воркер без опции отвечает на kind 4 `unknown worker input kind` (fail closed).
- В Rust kind 4 **пока нет нигде** (отложено намеренно). Вариант `WorkerInputKindV1::MlirLlvmText`
  и флаг `cargo fe2o3 engineering hsaco --lowering mlir` не добавлены. Причины:
  - `WorkerInputKindV1` (`fe2o3-hsaco-finalize/src/worker_protocol.rs:25`) — общий тип
    production-протокола и квитанций. Исчерпывающий `match` по нему всего один:
    `provider_kind` в `cargo-fe2o3/src/engineering_hsaco/support.rs:205-211`. Остальные места
    нового варианта не требуют:
    - декодеры байта kind в `fe2o3-hsaco-finalize/src` (`worker_protocol_v2.rs:1385-1391`,
      `worker_v3_compact_finalizer_replay.rs:1536-1545`, `first_build_worker_v3.rs:1740-1745`)
      заканчиваются веткой `_ =>` с ошибкой и уже отвергают 4;
    - `request_construction.rs:105-108` сопоставляет `CompilerModuleKindV1`, а не
      `WorkerInputKindV1`;
    - `cargo-fe2o3/src/engineering_hsaco.rs:563-572` разбирает строку `--provider` с веткой `_`.

    Технически вариант добавить дёшево. Но он сразу стал бы представимым в production-типах:
    его можно было бы закодировать в запрос и в квитанцию (`kind as u8` в
    `worker_v3_compact_finalizer_replay.rs:1330`). Решение о том, принимают ли это
    production-декодеры, относится к фазе 4, а не к инженерному пути;
  - свежий лоуэринг KIR → compiler module в `rustc-codegen-fe2o3` есть только под
    `#[cfg(test)]`: `construct_inert_compiler_module_text_v1` и `..._for_target_v1`
    (`kernel_ir_codegen.rs:237-245`). Вне тестов код только удерживает текст, уже построенный
    production-транзакцией (`retain_production_compiler_module_text_v1`, `:311`), так что флагу
    cargo пока нечего вызывать.
  Сейчас MLIR-путь проверяется только тестами (см. фазу 2). В `CompilerModuleKindV1`
  (production handoff) нового варианта нет и не будет до фазы 4.

### 3.2 MLIR-стадия воркера (`src/WorkerMlirInput.cpp`, реализовано)

1. Байты ASCII без NUL (как для `LlvmTextIr`), иначе `noncanonical textual MLIR bytes`.
2. `MLIRContext(MLIRContext::Threading::DISABLED)`: песочница запрещает создавать потоки
   (`RLIMIT_NPROC=0`, проверка `processCreationIsDenied`).
3. `DialectRegistry`: только `builtin`, `llvm`, `rocdl` и их LLVM IR translation interfaces;
   `allowUnregisteredDialects(false)`.
4. `ScopedDiagnosticHandler` собирает диагностику MLIR с метками severity: сначала ошибки,
   потом предупреждения и заметки, чтобы поток warning не вытеснил ошибку. Любое сообщение
   стадии, включая отказы обхода с именами из входа, ограничено `MaxDiagnosticBytes − 256`
   байтами и при обрезке получает маркер `<truncated>`. Без этого пайплайн молча режет
   диагностику по `MaxDiagnosticBytes`.
5. Парсинг через `SourceMgr` из собственной копии байтов: лексер MLIR читает NUL-sentinel за
   концом буфера, а в байтах запроса его нет. Верификация включена. Затем обход:
   - каждая операция должна быть из `llvm`/`rocdl`, а `builtin.module` допускается только
     корневым;
   - **незарегистрированные** операции `llvm.*`/`rocdl.*` запрещены: оба диалекта разрешают
     unknown ops, и такие операции доходят до транслятора, минуя verifier;
   - `llvm.named_metadata` запрещена: через неё можно подделать `llvm.dbg.cu` или
     `llvm.module.flags`, которые транслятор и контракт модуля читают без проверки (транслятор
     падал на произвольном `llvm.dbg.cu`). Для module flags есть типизированная операция;
   - discardable-атрибуты запрещены везде, кроме корневых `llvm.target_triple` (только
     `StringAttr`: транслятор делает непроверенный `cast<StringAttr>` и падает на assert) и
     `llvm.data_layout`;
   - символы с NUL в имени запрещены;
   - `ConstantRange` с равными границами (`<i32, 5, 5>`) запрещён везде, включая
     `rocdl.* range <...>`: в LLVM это assert;
   - inherent `alignment` (load/store/alloca/func/global) должен быть степенью двойки
     ≤ `Value::MaximumAlignment` (0 означает «нет»): транслятор передаёт его в `Align`
     без проверки;
   - атрибуты параметров и результатов (`ArgAndResultAttrsOpInterface`, в том числе
     **call-site** в типе `llvm.call`) проходят только из allowlist memory-фактов: `align`,
     `noalias`, `noundef`, `nonnull`, `nocapture`, `nofree`, `readonly`, `writeonly`,
     `readnone`, `dereferenceable(_or_null)`, `range`. Сверх MLIR-verifier проверяется ещё:
     unit-атрибуты без значения; `align` — степень двойки в пределах LLVM;
     `dereferenceable` > 0; `range` — именно `#llvm.constant_range` той же ширины, что и
     тип. Остальное транслятор **молча выбрасывает**: неизвестные `llvm.*` уходят в hook,
     который только пишет warning;
   - `llvm.inline_asm`: без `operand_attrs` (elementtype для `=*m` и т. п.), только LLVM-типы
     операндов и результата, constraints проходят `InlineAsm::verify` против собранного
     `FunctionType`. Иначе транслятор строит невалидный `InlineAsm` и падает;
   - имена аргументов: MLIR не хранит SSA-имена, а HSACO публикует их как `.name` в
     метаданных аргументов. Поэтому имя берётся из `loc("name")` аргумента entry-блока.
     Отказом считаются: пустое, длиннее 1024 байт (LLVM молча обрезал бы), с NUL, повторное,
     вложенные `NameLoc` и `NameLoc` внутри другой локации (`fused[...]`; иначе аргумент
     молча остался бы без имени). Переименование аргумента при коллизии в LLVM тоже отказ.
6. **Layout admission до трансляции.** Колбэк пайплайна (тот же, что у LLParser для
   текстового IR) получает корневые triple и layout и возвращает допущенный layout. Он
   записывается в корневой `llvm.data_layout` *до* `translateModuleToLLVMIR`, потому что
   транслятор берёт выравнивания load/store/alloca без `alignment` из стартового layout. Под
   layout по умолчанию i64 выравнен на 4, а `ptr addrspace(3)` на 8; у AMDGPU наоборот, 8 и 4.
   В LDS это меняет `ds_read_b64` на `ds_read2_b32`.
7. `translateModuleToLLVMIR(..., disableVerification=true, <пустая VFS>)`. Собственный verifier
   транслятора пишет неограниченный текст в `llvm::errs()` и возвращает пустую диагностику.
   Модуль всё равно верифицируется общим путём с ограниченным выводом.
8. Удаляется `Debug Info Version`, если модуль без отладочной информации (транслятор добавляет
   его безусловно).
9. Дальше общий путь, как для текстового IR: `setAndCheckModuleContract`
   (triple, xnack/sramecc flags, `code_object_version`), `verifyModule`, контракт запуска ядра
   V2, opt, codegen, LLD, инспекция HSACO.

Протокол: kind 4 принимается только в слоте compiler module запроса V2. В V1-входах и в
провайдерах V2 сборка с MLIR отвечает `MLIR input is admitted only as the V2 compiler module`,
сборка без MLIR отвечает `unknown worker input kind`.

**Ловушки транслятора, найденные в фазе 1:**
- module flags, записанные **атрибутом** модуля `llvm.module_flags = [...]`, транслятор молча
  игнорирует. Работает только операция `llvm.module_flags [...]` (атрибут запрещён правилом
  discardable-атрибутов);
- неверифицируемый на уровне LLVM модуль (например, `amdgpu-flat-work-group-size` не равен
  произведению `reqd_work_group_size`) при включённой верификации транслятора даёт пустую
  ошибку и мусор в stderr (см. п. 7).

**Известное ограничение, не MLIR:** malformed module flags в *текстовом* LLVM IR (например,
`amdgpu.xnack` со строковым значением) приводят к abort в `setAndCheckModuleContract`, а не к
отказу. Проблема существовала и до MLIR-пути и в этой ветке не исправляется. MLIR-путь её не
открывает: `llvm.module_flags` — типизированная операция, а `llvm.named_metadata` запрещена.

### 3.3 Rust-эмиттер `crates/fe2o3-amdgcn-model/src/lowering/mlir_text_v1.rs`

- Вход: тот же `Module` + `KernelId` после `verify_module`, та же модель `LoweringTarget`.
- Выход: `String` с MLIR в `llvm`-диалекте. Печать детерминированная: SSA-имена по
  ординалам KIR, порядок блоков как в KIR.
- Подмножество фазы 2: G1 math-free, то есть целочисленная и float-арифметика без OCML,
  сравнения, select, загрузки и сохранения global/LDS/kernarg, workitem/workgroup id, размеры
  через dispatch ptr, ветвления и циклы (phi через аргументы блоков), барьер. Всё остальное даёт
  `LoweringErrors` с кодом `LoweringDiagnosticCode::UnsupportedOperation` (helper
  `unsupported`, `mlir_text_v1.rs:70`), `UnsupportedType` (`:202`) или `UnsupportedTerminator`
  (`:847`). Это fail closed, без молчаливых пропусков.
- Таблица соответствий (проверена ридером по исходникам LLVM 24; 51 exact, 11 equivalent,
  3 differs):

| Конструкция `lowering.rs` | MLIR |
|---|---|
| target triple / datalayout | `module attributes {llvm.target_triple = "...", llvm.data_layout = "<дословно>"}` (строка, не DLTI). Layout для обоих таргетов дословно `GFX942_XNACK_MINUS_DATA_LAYOUT`, как у текстового пути; `p10`-`p15` добавляет `AdmitLayout` воркера (для `mlir-translate` в тестах layout допускается как есть) |
| kernel CC | `llvm.func amdgpu_kernelcc @k(...)` |
| имена параметров `%argN`, `%argN.data`, `%argN.len` | `%x: !llvm.ptr<1> loc("argN")` у аргумента entry-блока (иначе HSACO теряет `.name`) |
| `reqd_work_group_size` | `reqd_work_group_size = array<i32: X, Y, Z>` |
| `"amdgpu-flat-work-group-size"="N,N"`, `"amdgpu-waves-per-eu"` | `passthrough = [["amdgpu-flat-work-group-size", "N,N"], ...]` |
| `nounwind`, `convergent` | `no_unwind`, `convergent` (unit-атрибуты) |
| target cpu / features | `target_cpu = "gfx942"`. `target_features = #llvm.target_features<["-wavefrontsize32", "+wavefrontsize64", "-xnack"]>` ставится **только** если у модуля, ядра или функции есть `TargetCapability::WaveWidth`; это зеркалит текстовый путь. Без него features нет вовсе, и воркер отвергает такое ядро (`has no wavefront contract`) на обоих путях |
| `load ... align 4`, `fmul contract`, `add nuw` | strict properties LLVM 24: `llvm.load %p {alignment = 4 : i64}` **отвергается**, только `llvm.load %p <alignment = 4>` (то же у store и атомиков); `llvm.fmul %a, %b fastmath<contract>`; `llvm.add %a, %b overflow<nuw>` |
| denormal mode | `denormal_fpenv = #llvm.denormal_fpenv<default_*/float_* = ieee>` на каждой функции. Это то, во что `UpgradeFunctionAttributes` превращает `"denormal-fp-math"` текстового пути; legacy-строка через passthrough была бы инертна |
| прочие FP-строки | `passthrough` с тем же набором `"unsafe-fp-math"="false"`, ..., `"fp-contract"="off"`, что у текстового пути |
| `llvm.amdgcn.*` intrinsics | `rocdl.workitem.id.*`/`rocdl.workgroup.id.*` или `llvm.call_intrinsic` (dispatch ptr); декларации создаются транслятором с каноническими атрибутами |
| `fptosi.sat`/`fptoui.sat` | `llvm.call_intrinsic "llvm.fptosi.sat.iN.f32"` и т. д. (не `arith.fptosi` и не `llvm.fptosi`, они не насыщающие). Допускается только источник `f32`, иначе отказ |
| `switch` | `llvm.switch %v : iN, ^default [ c: ^bb, ... ]`. Значения case печатаются **знаковыми** литералами ширины селектора (`u64::MAX` при i64 становится `-1`), порядок как в KIR, пустой список печатается как `[]` |
| барьер | только физическая политика `s_barrier` (`requires_physical_workgroup_barrier`): fence до и после по ordering плюс `llvm.inline_asm has_side_effects "s_barrier"`. Relaxed-барьер означает отказ |
| constrained FP (`round.tonearest`, `fpexcept.ignore`) | **фаза 3** (StrictFloat-таргет сейчас отвергается). План: typed op добавляет `strictfp` на call site, поэтому `llvm.call_intrinsic` с `metadata`-аргументами |
| xnack | **ничего**: module flags ставит воркер (см. 2.1) |

**Что эмиттер отвергает (fail closed, `UnsupportedOperation` с префиксом `MLIR LLVM V1 emitter`):**
- таргеты `Baseline` и `Gfx942StrictFloatV1`;
- semantic anchors, checked-load grouping, ordered region/program/composition, debug line;
- native bf16, float-support helpers (узкие float, OCML), AMDGPU diagnostics, memory intrinsics;
- операции `Wave` и `Matrix`, relaxed-барьеры;
- функцию, чей таргет отличается от таргета модуля.

Кроме того, публичные функции `lower_compiler_module_to_gfx9{42,50}_xnack_minus_mlir_v1` отвергают
модуль, привязанный к другому таргету.

**Что неприменимо сейчас:**
- launch policies (`amdgpu-waves-per-eu`) достижимы только через `Gfx942StrictFloatV1`, а он
  отвергается. Код печати в эмиттере есть, но на этом подмножестве он мёртв;
- wave32: CDNA (gfx942/gfx950) только wave64. С `+wavefrontsize32` `opt` удаляет ядро, так что
  дифференциал для wave32 бессмыслен. Ветка печати есть только ради зеркальности с текстом.

Байтового совпадения с текстом `lowering.rs` нет и не будет: другие имена аргументов, нумерация
attribute groups и канонические атрибуты деклараций. К тому же `mlir-translate` строит IR через
`IRBuilder`, а он сворачивает операции над константами (например, `llvm.add` двух
`llvm.mlir.constant`) прямо при построении, тогда как LLParser сохраняет их как есть. Поэтому даже IR **до** оптимизации
различается побайтно. Существующие byte-golden для MLIR не годятся. Сравнение идёт на трёх
уровнях (подробно в фазе 2):
- канонический IR после bitcode-апгрейда (Rust; самая строгая проверка);
- нормализованный O3-ассемблер (Rust);
- байты HSACO через воркер (корпусный режим).

## 4. Фазы

### Фаза 1: MLIR-вход воркера (C++)
1. CMake: `FE2O3_WORKER_ENABLE_MLIR_INPUT`, `find_package(MLIR CONFIG)` с проверкой
   `MLIR_VERSION == LLVM_PACKAGE_VERSION`, линковка `MLIRLLVMDialect`, `MLIRROCDLDialect`,
   `MLIRParser`, `MLIRTargetLLVMIRExport`, `MLIRLLVMToLLVMIRTranslation`,
   `MLIRROCDLToLLVMIRTranslation`, `MLIRBuiltinToLLVMIRTranslation` (статические архивы поверх
   `libLLVM.so`; запрет на смешение с компонентными архивами LLVM, см. комментарий в CMake).
2. `InputKind::MlirLlvmText = 4` в `WorkerProtocol.{h,cpp}` под `#if FE2O3_WORKER_MLIR_INPUT`.
3. `src/WorkerMlirInput.cpp`: `Expected<std::unique_ptr<Module>> translateMlirInput(...)`.
4. `parseModuleInput` диспатчит kind 4 в MLIR-стадию, дальше общий код.
5. Тесты `tests/MlirInputTests.cpp`:
   - MLIR-фикстура, эквивалентная существующей текстовой (fill/saxpy), даёт **тот же
     `.text`** и те же `amdhsa`-метаданные, что и текстовый путь;
   - отказы: чужой диалект (`arith`), `gpu.module`, несовпадающий datalayout,
     конфликтующий `amdgpu.xnack`, невалидный MLIR (диагностика ограничена по длине);
   - работа под запретом потоков (`RLIMIT_NPROC=0`).

**Статус фазы 1: реализовано, прошло adversarial-ревью, находки исправлены; не закоммичено.**
- Тесты `fe2o3-worker-mlir-input-tests`:
  - **эквивалентность HSACO с текстовым LLVM IR, байт в байт:**
    - два ядра (vecadd и strided scale с циклом и phi), 6488 байт. Проверены три варианта
      заголовка: машинный triple и layout, без triple и layout, legacy `amdgcn-amd-amdhsa` с
      layout LLVM 22. Повторный прогон даёт те же байты;
    - два LDS-ядра с load/store **без** выравнивания (i64 и `ptr addrspace(3)`) и
      internal-переменной в LDS, те же три заголовка;
    - матрица опций O0..O3 × {strip+verify, без них}.
  - **прямая трансляция:** module flags, triple, CC, атрибуты, имена аргументов. Разрешённые
    формы (range у параметра и у `rocdl.workitem.id.x`, `inline_asm` с `=v` и пустыми
    constraints, `align`/`dereferenceable`, имя ровно 1024 байта) транслируются с этими
    фактами. Layout применяется до трансляции: выравнивания 8/4/8/4 и контрольный прогон без
    колбэка, который их меняет;
  - **45 отказов на стадии `InputValidation`, каждый со своим текстом:**
    - диалекты и операции: чужие диалекты, вложенный `builtin.module`,
      `unrealized_conversion_cast`, незарегистрированные `llvm.*`, `llvm.named_metadata`
      (`llvm.dbg.cu`, `llvm.module.flags`);
    - атрибуты: `llvm.module_flags` атрибутом, нестроковый triple, `rocdl.kernel`; range
      (не range, unit, равные границы, чужая ширина; у параметра, у call-site и у
      `rocdl`-операции); `alignment` 3 и 2^33 (load, alloca, func); `llvm.align` 3 и 0;
      `noalias` со значением на call-site;
    - `inline_asm`: `operand_attrs`, несовпадающие constraints;
    - имена: NUL, 1025 байт, `fused`, вложенные `NameLoc`, повтор, NUL в символе;
    - контракт модуля: чужой layout, x86 triple, конфликт `amdgpu.xnack`, flat size не
      равен `reqd_work_group_size`;
    - текст: невалидный, обрезанный payload, не-ASCII;
    - диагностика: длинные отказы на parse-пути и после обхода (`<truncated>`), ошибка
      после 8 длинных warning (ошибка идёт первой).
  - декодер: kind 4 принимается только как compiler module V2; в V1 и в провайдерах V2 отказ
    в обеих сборках (контрольный V1-запрос с kind 3 падает по другой причине);
  - сборка без MLIR отвечает `MLIR input refused by this worker build`.
- Отдельный ctest `fe2o3-worker-mlir-input-thread-free` (`--thread-free`):
  - fork выполняется до создания любого состояния LLVM/MLIR. Потомок ставит все пять лимитов
    authenticated machine-effect режима `main.cpp` (`RLIMIT_AS` 4 GiB, `RLIMIT_DATA`
    2 GiB, `RLIMIT_FSIZE` 16 MiB, `RLIMIT_CORE` 0, `RLIMIT_NPROC` 0). Обычный режим линковки
    лимитов не ставит;
  - потомок проверяет, что `pthread_create` падает. Затем компилирует и отдаёт HSACO через
    pipe, а родитель сравнивает его с текстовым путём;
  - код 77 = skip, если `RLIMIT_NPROC` не запрещает потоки или сборка без MLIR.
- Мутационная проверка тестов:
  - перенос layout на после трансляции ловят и прямой тест, и сравнение HSACO LDS-ядер;
  - warning перед ошибками ловит тест с 8 warning.
- ctest: 9/9 с `FE2O3_WORKER_ENABLE_MLIR_INPUT=ON`; без неё 8/8 и 1 skip (thread-free).
- Размер воркера: 39.3 МБ с MLIR против 5.3 МБ без. Динамически подключается только
  `libLLVM.so.24.0git`; MLIR линкуется статическими архивами, LLVM дважды не попадает.

### Фаза 2: Rust-эмиттер + дифференциальные тесты

**Статус: реализовано и зелёное; не закоммичено.**

1. **Эмиттер** `src/lowering/mlir_text_v1.rs` (около 2250 строк). Он потребляет то же
   preflight-состояние `FunctionLowerer`, что и текстовый рендерер, через общий
   `lower_compiler_module_with_emitter_v1(..., CompilerModuleEmitterV1::MlirLlvmV1)`. Поэтому
   выбор инструкций зеркалит `lowering.rs` один к одному, а всё, что вне подмножества 3.3,
   отвергается до печати текста. Публичные функции:
   `lower_compiler_module_to_gfx942_xnack_minus_mlir_v1` и `..._gfx950_...`. Ни квитанции, ни
   дайджесты, ни lineage к этим байтам не привязаны.
2. **Общий harness** `tests/support/mlir_differential_v1.rs`. Подключается через `#[path]` в 8
   тестовых крейтах; подмодули `compiler_module/*` и `lowering/mlir_text_v1.rs` берут его у
   родителя. Так каждый переиспользует свои построители модулей. `assert_equivalent` делает
   следующее:
   - оба рендерера должны принять модуль;
   - структурная проверка MLIR без LLVM:
     - точный заголовок модуля: triple `amdgcn-amd-amdhsa` (legacy; LLVM 24 предупреждает
       о triple без subarch, воркер принимает обе формы) и layout;
     - сбалансированные скобки;
     - запрещённые конструкции: `llvm.named_metadata`, `llvm.module_flags`,
       `llvm.intr.fpto?i.sat`, `{alignment = ` вне `llvm.alloca`;
     - у каждого определения есть `target_cpu`, `no_unwind` и `denormal_fpenv`, у ядра есть
       `reqd_work_group_size` и flat size;
     - SSA-имена уникальны, метки блоков выровнены;
     - **имена параметров по порядку** (`assert_parameter_names`). Набор определений и
       список имён параметров каждого из них должны совпасть с `define` текстового пути,
       включая порядок. Каждый параметр MLIR обязан иметь вид `%L: ... loc("L")`, то есть
       `loc` висит на SSA-значении с тем же именем. Именно из `loc` воркер восстанавливает имя
       LLVM-аргумента, а HSACO публикует его как `.name`. Прежняя проверка «каждый `loc`
       называет какой-то параметр» перестановку не ловила;
   - **дифференциал**, если `FE2O3_LLVM_BIN` указывает на каталог с `mlir-translate`, `opt` и
     `llc`:
     1. `mlir-translate --mlir-to-llvmir`;
     2. **апгрейд** обоих модулей через bitcode, как в пайплайне воркера: `opt -passes=verify
        X -o X.bc`, затем `opt -passes=verify -S`. Без этого строковые
        `"denormal-fp-math-f32"` текстового пути не превращаются в `denormal_fpenv`: в LLVM 24
        этот upgrade делает только BitcodeReader;
     3. **канонический IR** (`canonical_ir`) обеих форм должен совпасть построчно. Канонизация:
        - named types, attribute groups (`#N`, атрибуты сортируются) и узлы `!N` (рекурсивно)
          подставляются на место ссылок: текст использует именованные struct-типы и свою
          нумерацию групп, MLIR литеральные типы;
        - параметры становятся `%aN`, значения `%vN`, блоки `%bbN`. Неявный слот безымянного
          entry-блока у `mlir-translate` (`%{число нумерованных параметров}`) становится
          `%bb0`;
        - `getelementptr` глобала со всеми нулевыми индексами (флаги
          `inbounds`/`nuw`/`nusw` допустимы) заменяется самим глобалом: `IRBuilder` такие
          GEP сворачивает, LLParser нет;
        - входящие пары `phi` сортируются по номеру блока: порядок у путей разный;
        - строки `target`, глобалы и **используемые** декларации сортируются; функции
          сохраняют порядок;
        - отбрасываются комментарии, `; ModuleID`/`source_filename`, именованные метаданные
          (module flags ставит воркер) и неиспользуемые декларации.

        Всё остальное сравнивается дословно: типы, выравнивания, ordering и syncscope
        атомиков, флаги `nuw`/`nsw`/fastmath, атрибуты функций, параметров и call-site;
     4. обе формы проходят `opt -O3` и `llc -O3 -mcpu=gfxNNN`. Нормализованный ассемблер
        (`normalize_assembly`) должен совпасть полностью. Берётся всё от первого `.text` до
        `.end_amdgpu_metadata` включительно: инструкции, директивы, `.amdhsa_kernel` и
        HSA-метаданные. Отбрасываются только комментарии после `;`, пустые строки,
        `.file`/`.ident` и позиционные `.name: argN` (`mlir-translate` теряет имена
        аргументов, воркер восстанавливает их из `loc`; это проверяет структурная часть);
     5. дескриптор ядра проверяется отдельным assert.

     При расхождении сообщение называет первую различающуюся строку обеих сторон и
     scratch-каталог (`text.canon`/`translated.canon`, промежуточные `.ll`/`.bc`/`.s`);
     `FE2O3_MLIR_KEEP=1` его сохраняет;
   - **без `FE2O3_LLVM_BIN`** выполняется только структурная часть. Каждый тестовый бинарник
     один раз пишет в stderr `note: FE2O3_LLVM_BIN is unset; the MLIR differential is skipped
     ...` (через `std::io::stderr()`, иначе libtest проглатывает вывод).
     `FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1` превращает пропуск в падение теста; так пропуск не
     может пройти незамеченным там, где дифференциал обязателен;
   - **CI** (`.github/workflows/ci.yml`, `ubuntu-24.04`, `apt-get install llvm`, то есть
     LLVM 18 без `mlir-translate`) `FE2O3_LLVM_BIN` не задаёт. Поэтому в CI MLIR-тесты
     выполняют только структурную часть и fail-closed проверки. Дифференциал и корпус
     воркера пока запускаются только локально с LLVM 24 (§6).

   **Мутационная проверка harness.** Adversarial-ревьюер сделал 29 мутаций MLIR-файлов
   корпуса: выравнивания load/store/alloca/атомиков/LDS, ordering и scope атомиков,
   `nounwind`, fp-атрибуты (`fp-contract`, `unsafe-fp-math`, denormal), порядок и имена `loc`,
   удалённый store, `fadd`→`fsub`, `ult`→`ule`, размер workgroup, чужой cpu и т. д. Из них
   3 файл не изменили, а 1 дала пустой файл (ловится тривиально, на заголовке модуля).
   Все 25 содержательных ловятся:
   - 19 каноническим IR;
   - 2 структурой (`cpu950`, `nounwind`);
   - 4 перестановки и переименования `loc` проверкой имён.

   `assert_fails_closed` требует, чтобы текстовый путь модуль принял, а MLIR отверг с
   диагностикой `MLIR LLVM V1 emitter` и ожидаемой причиной.
3. **Корпус: 33 модуля × 2 таргета = 66 пар.** Состав:
   - `lowering/mlir_text_v1.rs`: fill, private pointer slot, phi loop, vecadd,
     g4 synchronization (LDS, барьеры, fence), scoped atomics, атомик через прямой глобальный
     указатель, raw/empty/integer switch, unreachable block;
   - `control_flow` (ветвления, integer match, вложенный цикл), `guarded_load` (в том числе
     volatile), `checked_binary`, `float_casts`, `unary`, `bitwise_binary`;
   - `compiler_module`: helpers, effective wave64, workgroup geometry в helper,
     `self_contained_compiler_module` (тот же модуль из двух ядер, но `public_adjust` зовёт
     `scale` вместо внешнего `external_bias`; это многоядерный модуль, который корпусный
     режим может слинковать); `counted_private_alloca_v1`, `ordinary_helper_results_v1` (pair, multiple return, loop,
     width 256), `shared_slice_helpers_v1`.

   Отдельные тесты проверяют:
   - знаковые case-литералы и `[]` для пустого switch;
   - отсутствие `llvm.module_flags`, `llvm.named_metadata` и `amdgpu.xnack` в выводе;
   - детерминизм и привязку к таргету (gfx950-модуль отвергается gfx942-функцией и наоборот);
   - fail closed: sqrt (OCML), Wave, gfx950 bf16 MFMA.
4. **E2E через воркер (корпусный режим).** `FE2O3_MLIR_DUMP_DIR=DIR` заставляет каждый
   `assert_equivalent` записать пару `{файл}-{имя}.{cpu}.ll` и `.mlir`. Запись идёт через
   rename, так что параллельные тесты безопасны, а различающийся дубль под тем же именем
   роняет тест. Затем `fe2o3-worker-mlir-input-tests --corpus DIR` прогоняет каждую пару через
   полный V2-пайплайн: текст как `LlvmTextIr`, MLIR как `MlirLlvmText`. Порядок проверок:
   - **сироты:** каждый `.mlir` обязан иметь `.ll`-партнёра и наоборот, а суффикс stem должен
     быть `.gfx942` или `.gfx950`;
   - **замыкание, для каждой пары** (`requireSameClosure`). MLIR транслируется собственной
     стадией воркера (`translateMlirLlvmTextInput`), и результат сравнивается с разобранным
     текстом. Должны совпасть:
     - экспорты, finals и число ядер;
     - у каждого определения linkage, calling convention, число аргументов и **имена
       аргументов по порядку** (восстановленные из `loc`);
     - число определений;
     - импорты. Правило импорта то же, что в `inspectModuleSymbols` воркера: используемая
       не-intrinsic декларация для линкера;
   - **линковка** пар без импортов. Требуется `post_link.check=metadata status=ok kernels=K`,
     после чего HSACO сравниваются **байт в байт**. Среди них есть многоядерный
     `self_contained_compiler_module`.

   Оговорки:
   - для дампа модуль без какого-либо `WaveWidth` получает wave64 на уровне модуля:
     воркер без wavefront-контракта отвергает ядро на обоих путях. Сам дифференциал в
     процессе теста идёт на модуле как есть;
   - пары с импортами проверяются только на замыкание, без линковки: воркер требует для
     импортов манифест, а корпус его не строит. Это `checked_all` (33 импорта
     `checked_sink_*`), `compiler_module` (`external_bias`) и `g4_synchronization`
     (динамическая LDS `__fe2o3_lds_fill_31`);
   - **найденное расхождение контракта символов** (не кода) у `g4_synchronization`.
     Динамическая LDS (`external addrspace(3) global [0 x float]`) в текстовом пути
     используется только мёртвым `getelementptr ... i32 0, i32 0`, и `inspectModuleSymbols`
     считает её импортом. `IRBuilder` в `mlir-translate` такой GEP сворачивает в сам глобал, и
     MLIR-модуль не содержит ни одного использования, то есть импорта нет. Машинный код
     одинаков: канонический IR в Rust совпадает, потому что нулевой GEP там заменяется
     глобалом. Корпусный режим разрешает **ровно** это различие: текстовый импорт, у которого
     все пользователи — мёртвые инструкции без побочных эффектов, может отсутствовать в MLIR.
     Такие случаи печатаются явно (`closure identical but for the textual import of ... by
     dead instructions only`) и считаются в итоговой строке. Любой другой лишний или
     пропавший импорт роняет тест. Для production это значит, что манифест импортов
     MLIR-модуля может оказаться короче текстового. Это нужно учесть в фазе 4, если манифест
     строится по тексту;
   - **сравнение байтов HSACO слабее канонического IR.** На тех же 25 мутациях воркер ловит
     10: 4 по именам параметров, 5 по различию HSACO (`c256to128`, `denorm`, `dropstore`,
     `fadd2fsub`, `ult2ule`) и 1 по отказу запроса (`cpu950`). Ещё 1 (`g4_body`) приходится
     на пару с импортом, которая не линкуется. Остальные 14 дают **идентичные байты**:
     выравнивания load/store/alloca/LDS/атомиков, ordering и scope атомиков, `nounwind`,
     `fp-contract`, `unsafe-fp-math`, implicit-аргументы, слот private-указателя. Для этих
     ядер и этой ревизии LLVM они на машинный код не влияют, но в другом контексте влияли бы.
     Поэтому авторитетная проверка эквивалентности — канонический IR в Rust, а корпус воркера
     добавляет к ней проверку собственной трансляции воркера, имён аргументов и полного
     V2-пайплайна.

   **Результат: `MLIR corpus passed: 60 identical (2 multi-kernel), 6 closure-only (2 with
   dead-use textual imports)`.** Это 66 пар: 60 дают идентичный HSACO, 6 пар с импортами
   (3 модуля × 2 таргета) прошли проверку замыкания, расхождений 0.
5. **Полный набор `cargo test -p fe2o3-amdgcn-model`** (с `FE2O3_LLVM_BIN` и LLVM 24 в
   `PATH`): 582 passed, 0 failed, 26 ignored. Это 566 прежних тестов и 16 новых MLIR-тестов.
6. **ctest воркера `mlir-input`** проходит в обеих сборках: с MLIR 2/2; без MLIR 2/2, причём
   thread-free пропускается с кодом 77.

**Чего в фазе 2 нет, и почему:**
- Rust `WorkerInputKindV1::MlirLlvmText` и флаг `cargo fe2o3 ... --lowering mlir` отложены (см. 3.1);
- KIR-симулятор как оракул не понадобился: на всём корпусе машинный код идентичен, так что
  эквивалентность по ISA строже симулятора. Прогоны на GPU остаются за пользователем;
- дифференциал и корпус воркера идут только на O3, как и production-путь. На O0 различия
  pre-opt IR (свёртка констант `IRBuilder`) могут дойти до машинного кода, поэтому O0 не
  сравнивается. Матрица O0..O3 есть только у рукописных фикстур фазы 1. Канонический IR
  сравнивается до оптимизации, так что он от уровня O не зависит;
- **дублирование таблиц syncscope оставлено намеренно.** `mlir_atomic_sync_scope` и
  `mlir_fence_sync_scope` (`mlir_text_v1.rs:336-353`) повторяют отображение
  `llvm_atomic_sync_scope` (`lowering.rs:9880`) и `emit_fence` (`lowering.rs:9849`) текстового
  пути. Общий helper
  потребовал бы менять production-эмиттер `lowering.rs`, а его поведение и байты в этой ветке
  не трогаются. Дрейф между копиями ловит канонический IR: syncscope сравнивается дословно,
  мутация `scope2` поймана. Объединять стоит в фазе 4 вместе с остальным production-ревью.

### Фаза 3: расширение покрытия
Строгий FP (constrained intrinsics через `call_intrinsic`) вместе с таргетом
`Gfx942StrictFloatV1` и, следовательно, launch policies; bf16/MFMA (`rocdl.mfma.*`), Wave-операции,
semantic anchors (`llvm.pseudoprobe` + `!llvm.pseudo_probe_desc`), debug line info (v17),
memory intrinsics, затем OCML (провайдеры воркера уже работают с `llvm::Module`, MLIR-путь их
не затрагивает) и gfx950 после профиля 2.3. Отдельно: манифест импортов в корпусном режиме
воркера, чтобы слинковать 6 closure-only пар (с учётом расхождения по мёртвой динамической LDS,
см. фазу 2, п. 4); job в CI с LLVM 24 и `FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1`, чтобы
дифференциал и корпус шли не только локально; Rust kind 4, когда появится нецелевой
(не `cfg(test)`) инженерный вызов compiler-module лоуэринга.

### Фаза 4: решение по production
Отдельное ревью: новый lineage, новые дайджесты `llvm_ir_sha256` (теперь уже для MLIR-текста),
новые квитанции, перепин production-констант на LLVM 24. До этой фазы production не меняется.

## 5. Риски

| Риск | Вероятность | Митигация |
|---|---|---|
| Эквивалентные, но не идентичные декларации intrinsics меняют codegen | низкая | дифф `.text` по всем G1-ядрам; canonical attrs ставит и LLParser (`Function.cpp:517`) |
| `strictfp` из typed constrained ops | средняя | только `call_intrinsic`; тест-отказ на появление `strictfp` |
| Транслятор добавляет `Debug Info Version` | точная | удаляется в воркере для модулей без DI; тест |
| Песочница воркера и потоки MLIR | точная | `Threading::DISABLED`; тест под `RLIMIT_NPROC=0` |
| Дрейф MLIR API между ревизиями LLVM 24 | средняя | версия MLIR = версия LLVM, проверка в CMake; MLIR в measurement preimage |
| Рост размера воркера (MLIR-архивы) | точная | `MaxWorkerExecutableBytes = 512 MiB` с запасом; замерить в фазе 1 |
| O0 inline asm (2.2) | точная для asm-путей | эмиттер не генерирует asm; правило для asm-путей |
| gfx950 на upstream LLVM (2.3) | точная | compile-only до нового профиля |
| Дифференциал молча не запускается (нет `FE2O3_LLVM_BIN`, как в CI) | точная для CI | заметка в stderr один раз на бинарник; `FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1` превращает пропуск в падение; CI-job с LLVM 24 в фазе 3 |
| Семантически разный IR даёт одинаковый HSACO и проходит корпус воркера | точная (14 из 25 мутаций) | авторитетна проверка канонического IR в Rust, она ловит все 25 |
| Манифест импортов MLIR короче текстового (мёртвый нулевой GEP динамической LDS) | точная для `g4_synchronization` | корпус допускает только это различие и печатает его; учесть при построении манифеста в фазе 4 |

## 6. Воспроизведение

```sh
# Воркер под LLVM 24 (+MLIR в фазе 1)
cmake -S tools/fe2o3-llvm-link-worker -B <build> -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/home/mrsidims/LLVM-work/build/lib/cmake/llvm \
  -DLLD_DIR=/home/mrsidims/LLVM-work/build/lib/cmake/lld \
  -DFE2O3_PINNED_LLVM_VERSION=24.0.0git \
  -DFE2O3_LLVM_BUILD_ID_FILE=<file с "24.0.0git-cea26e47..."> \
  -DFE2O3_EXPECTED_LLVM_BUILD_ID=24.0.0git-cea26e4798f7899d9c40259aad6f793038a1f01f \
  [-DFE2O3_WORKER_ENABLE_MLIR_INPUT=ON -DMLIR_DIR=/home/mrsidims/LLVM-work/build/lib/cmake/mlir]
cmake --build <build> -j8 && ctest --test-dir <build> --output-on-failure

# Rust-тесты с инструментами LLVM 24 (без PATH 12 старых тестов падают: "llvm-as is required")
export PATH=/home/mrsidims/LLVM-work/build/bin:$PATH
cargo test -p fe2o3-amdgcn-model --locked

# + дифференциал MLIR vs текст: канонический IR и O3-ассемблер (FE2O3_MLIR_KEEP=1 оставляет
# scratch-каталоги; FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1 роняет тест, если FE2O3_LLVM_BIN не задан)
FE2O3_LLVM_BIN=/home/mrsidims/LLVM-work/build/bin FE2O3_REQUIRE_MLIR_DIFFERENTIAL=1 \
  cargo test -p fe2o3-amdgcn-model --locked --tests mlir_v1

# E2E: дамп корпуса, затем оба пути через воркер с MLIR, сравнение байтов HSACO
# (каталог должен быть пустым; FE2O3_MLIR_TEST_DUMP_DIR=<dir> сохраняет несовпавшие объекты)
rm -rf <corpus> && mkdir <corpus>
FE2O3_MLIR_DUMP_DIR=<corpus> cargo test -p fe2o3-amdgcn-model --locked --tests mlir_v1
<mlir-build>/fe2o3-worker-mlir-input-tests --corpus <corpus>
# ожидается: "MLIR corpus passed: 60 identical (2 multi-kernel), 6 closure-only (2 with dead-use textual imports)"
```
