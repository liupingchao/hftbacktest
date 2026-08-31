# 0831T001 Implementation Candidate Round 6

执行线程：
- 业务线程

任务ID：
- 0831T001

状态：
- 执行中

日期：
- 2026-08-31

candidate basis：
- round 5 readiness failure commit：
  `ed7dc0c1`
- reviewed implementation commit：
  `a5ea995e17fd1773b77275db986b57d4517a332c`

scope：
- 只关闭 round 5 exact-commit preflight 发现的 readiness tree hash
  ordering mismatch。
- 不修改 frozen surface、truth、task、plan、模型逻辑、formal 状态机或
  round 5 已关闭的 recovery/controller findings。

implementation：
1. `readiness_projection_manifest()` 现在先按 ASCII path order 排序
   `projection_files`，再构造 `{path,size_bytes,sha256}` rows。
2. `tree_sha256` 继续使用 canonical compact sorted-key ASCII JSON，
   其完整 preimage 语义现在与 frozen surface contract 一致。
3. 新增乱序 contract 回归：输入 `["z.txt","a.txt"]`，要求 rows 与 tree
   hash 均按 `["a.txt","z.txt"]` 计算。

verification：
- targeted ordering 与 round 4 closure tests：
  `10 passed, 97 deselected in 0.16s`。
- focused pytest：
  `107 passed in 93.45s`。
- expanded accepted predecessor regression：
  `410 passed, 1 skipped in 104.79s`。
- ruff check、ruff format check、`git diff --check`：通过。
- historical cache / future market outcome：未访问。
- claim、controller、formal root、receipt 与 task tags：未创建。

remaining gate：
- 本候选必须提交后，在 primary 与 fresh detached exact commit 上重新
  完成 readiness。
- 必须逐文件 byte-compare，并独立按 frozen ASCII path order 重算 tree
  SHA256。
- exact-commit readiness 通过后，仍需 independent implementation
  readiness review；其 PASS 之前 formal 继续锁定。

提交信息：
- `implementation: checkpoint 0831T001 Q0 candidate round 6`
