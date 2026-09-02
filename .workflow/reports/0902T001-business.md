# 0902T001 Business Report

执行线程：
- Target Project Argv Contract Repair 业务线程

任务ID：
- 0902T001

状态：
- 待验收

日期：
- 2026-09-02

是否进行QA验收：
- 是

QA说明：
- 无

## Authority

- scope rebaseline commit：
  `d6f5147f4fd8d98b1a15d4cd90706f015fcc033a`
- controller boundary commit：
  `45514a9a380552fe03c1a6c5fbdb7a7bf7585d86`
- target base commit：
  `1051f2b29059e6b7465fe8051de01072f9ff7e19`
- target task registration commit：
  `cc8c21f987497bd0f96a64183574e98b6db7d3e9`
- implementation commit：
  `5eed10e59dcab91657bd332ca2b94ba3d2b7476b`

## Files

- `examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification.py`
- `examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier.py`
- `examples/hyperliquid/test_skhynix_trade_led_depth_follower_q0_pipeline_qualification.py`

No Workflow Kit, schema, scientific logic, surface contract, old claim, old
receipt or business-output file was modified.

## Action

1. Preserved every frozen complete executable command:

   ```text
   exec_argv = [python_executable, script_path, ...args]
   ```

2. Added one fail-closed derivation used by the outer driver, producer,
   recovery driver and terminal verifier:

   ```text
   program_argv = exec_argv[1:]
   observed Python sys.argv == program_argv
   observed Python sys.argv != exec_argv
   observed sys.executable == exec_argv[0]
   ```

3. Corrected armed-claim verification to reconstruct the observed complete
   command as `[sys.executable, *sys.argv]`; it no longer compares the claim's
   complete command with program-only `sys.argv`.

4. Added `--qualify-argv-contract`, an effect-free mode that reads and
   verifies runtime bytes, script bytes, cwd and argv identity, rejects
   `shell = true`, and emits evidence only to stdout.

5. Added regression and hostile coverage for:
   - the exact former `SOURCE_ROOT_NOT_CLOSED:argv` behavior;
   - derived `program_argv` success;
   - missing or drifted interpreter path/bytes;
   - missing, non-regular or drifted script bytes;
   - argv and cwd drift;
   - `shell = true` and shell command strings;
   - all four Python process boundaries;
   - zero output from effect-free qualification.

## Frozen Bytes

```text
runtime_path =
  /Users/liu/.local/conda/bin/python
runtime_resolved_path =
  /Users/liu/.local/conda/bin/python3.13
runtime_sha256 =
  333e66ec89afec4a6295f1afb6c50da4d9f5629ed5c3f831e9868bf8c5479b9f
runner_sha256 =
  efdaca45419b1e87573be68e3ec0bd398a344d8cb39194cb3c95a514811c4dc6
verifier_sha256 =
  e680b2b8413cac900a36419aef662dbff700b7fa43af864af5d8adefb5e9eced
tests_sha256 =
  4a1b763f62ff00f7e667a40ca2cf263bcab4708a7cd2e28bff50819b33928601
cwd =
  /Users/liu/Documents/hftbacktest-0902t001-argv-contract-repair
shell =
  false
```

The effect-free result reported:

```text
business_execution = false
effectful_outputs = false
successor_q0 = ABSENT
observed_sys_argv = program_argv
```

## Verify

- argv-focused regression:
  `16 passed, 132 deselected`.
- applicable current focused suite:
  `147 passed, 1 deselected`.
- expanded accepted predecessor suite:
  `410 passed, 1 skipped`.
- exact original implementation baseline at
  `879a763944e6b8052333b6102a2f940e18a0f664`:
  `130 passed`.
- ruff check and format check:
  passed.
- Python compile:
  passed.
- runner and verifier `--help`:
  passed without side effects.
- `git diff --check` and `git show --check`:
  passed.

The one current focused deselection is
`test_development_formal_pipeline_replays_all_registered_negatives`.
It is not an argv repair failure. Exact base commit `1051f2b...f7e19`
already changes `.workflow/tasks/0831T001.md` from `执行中` to `未通过`,
so its SHA256 is
`79142da5690d3611f2157accb03bf6f2b9dd84256b7ead55b4e0d1e00d0914f0`
rather than the frozen implementation SHA256
`19585d2501994535eca3d462b62860be76c54cc1196609ce9fd9a788255b138b`.
The unchanged base therefore deterministically fails that old development
formal test at `AUTHORITY_BINDING:.workflow/tasks/0831T001.md`. The exact
implementation commit's complete original suite passes `130/130`.

## Boundary

- `0902T001` armed claim: absent.
- `0902T001` claimed claim: absent.
- `0902T001` consumption receipt: absent.
- `0902T001` terminal receipt: absent.
- `0902T001` baseline/output root: absent.
- successor controller ref: absent.
- business/scientific execution: false.
- historical cache and future outcome access: none.
- old `0831T001` armed claim remains unchanged at SHA256
  `8a6b7f1078f2b6540771a5ea5144745de5893f2cdddc2132206892a1c5d501fc`.
- successor Q0 remains unregistered and locked.

## Done

- The target-local argv defect is repaired and committed.
- The exact failure is reproduced by the former comparison.
- The corrected contract passes effect-free qualification.
- Runtime, script, argv, cwd and shell evidence is frozen above.
- No business task or Q0 was registered or executed.
- The task is ready for independent read-only QA.

blockers：
- 无

commit：
- `5eed10e59dcab91657bd332ca2b94ba3d2b7476b`

提交信息：
- `fix: validate Python program argv`
