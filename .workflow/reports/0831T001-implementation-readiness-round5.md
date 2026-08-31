# 0831T001 Implementation Readiness Round 5

执行线程：
- 总控 exact-commit readiness preflight

任务ID：
- 0831T001

状态：
- 未通过

日期：
- 2026-08-31

是否进行QA验收：
- 否

reviewed commit：
- `a5ea995e17fd1773b77275db986b57d4517a332c`

severity counts：
- P0：0
- P1：1
- P2：0
- P3：0

finding：
1. P1：`readiness_projection_manifest()` 未按 surface contract 冻结的
   `tree_hash_preimage` 语义计算 readiness tree hash。Contract 要求
   `{path,size_bytes,sha256}` rows 按 ASCII path order 排序后做 canonical
   compact JSON SHA256；实现按 `projection_files` 注册顺序直接计算。
   当前注册顺序不是 ASCII path order，因此 runner 输出
   `763e9a0d1dbbb49f7baefd5e3414ffd355ddd95bc5ba190c344e74beb50e34f3`，
   独立按 contract 重算为
   `05db53ca596a99d845bd7b4e5a6f500c3cc6a5355958eb5ebaa9fae726cb57b7`。

verify：
- primary 与 fresh detached readiness 均完成 `57` feature calls，生成
  `37` 个注册 projection files。
- 两个 projection 目录逐文件 byte-identical，`diff -qr` exit `0`。
- 两个 runner 均报告相同的注册顺序 tree SHA256：
  `763e9a0d1dbbb49f7baefd5e3414ffd355ddd95bc5ba190c344e74beb50e34f3`。
- 独立 Ruby JSON/SHA256 重算证明：
  `registered_order_sha256=763e9a0d...e34f3`，
  `ascii_path_order_sha256=05db53ca...b57b7`，
  `same_order=false`。
- focused pytest：`106 passed in 91.87s`。
- expanded accepted predecessor regression：
  `410 passed, 1 skipped in 98.28s`。
- ruff、format、py_compile、`git diff --check`：通过。
- 主 worktree 与 detached readiness worktree 均指向 exact reviewed
  commit；claim、controller、formal root、receipt 与 task tags 不存在。
- historical cache / future market outcome：未访问。

结论：
- `FAIL`
- Round 5 在 independent reviewer 之前即被 exact-commit preflight
  拒绝；不产生独立 PASS、arming、controller、formal root 或 task tag。
- 下一候选只允许修复 readiness row 的 ASCII path sorting，并增加乱序
  contract regression test；随后重新执行完整 exact-commit readiness 和
  独立审计。

提交信息：
- `review: record 0831T001 Q0 readiness round 5 failure`
