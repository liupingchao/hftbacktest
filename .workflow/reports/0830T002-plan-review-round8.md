# 0830T002 Hostile Plan Review Round 8

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `dff5792d838d231d9c5d8f6637a219a10d20945e`
- idea SHA256
  `96d067722b28f53eed803e1e5c4f62a076a5bb8c166333aa253248df8fe5a888`
- plan SHA256
  `4dbcbf9af525b850a2b5b9f176163812a353d797a4309dcd9699e7c43f00427f`
- task SHA256
  `0fa740545e87cf073143e01502b9f2da9965de65893287d6a26858312fcb86a6`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 8 只修改 docs/task/workflow 记录，未修改 runner/tests。
- `git diff --check e9b8c521..dff5792d` 通过。
- Local `origin` fetch/push URL 均为计划注册的
  `git@github.com:liupingchao/hftbacktest.git`。

## Severity Summary

- P0: 0
- P1: 1
- P2: 2
- P3: 0

## Round 7 Closure Matrix

| Round 7 finding | Round 8 status | 结论 |
|---|---|---|
| P1-1 external ledger可被普通write credential改写，但threat model只排除admin | CLOSED | threat model现已明确排除任何持有remote write credential且偏离exact two-push protocol的actor，并把该行为定义为study invalidation |
| P1-2 IPC envelope未绑定actual payload/single-frame/EOF | PARTIAL | canonical frame、payload table、payload/frame hash、single `send_bytes`和EOF规则已写入；但持久化schema没有loader/detector独立观测，terminal verifier无法复核transient channel事实 |
| P2-1 remote/ref transition不唯一 | PARTIAL | exact URL、ref、push/ls-remote argv、absence/equality规则及verifier final observation已增加；`git push --porcelain`输出与full old/new tuple的authority仍未定义 |
| P2-2 FieldAccess、feature count、phase matrix、audit normalization不完整 | CLOSED | per-call 12-row hash、四方feature count equality、phase/path/caller matrix以及native `open` normalization均已唯一冻结 |

## Findings

### P1-1 Single-frame IPC has no independent sender/receiver evidence closure

位置：
- execution plan `:938-947`
- execution plan `:1185-1195`
- execution plan `:1391-1419`
- execution plan `:1758-1761`

问题：
- Revision 8 已定义合理的wire contract：
  - canonical header；
  - actual payload bytes及SHA/length；
  - exact one-frame `send_bytes`；
  - detector table/hash/length validation；
  - EOF、no second frame和no unused byte。
- 但持久化 `FeatureCall` 只有一组：

```text
ipc_envelope_sha256
ipc_frame_sha256
ipc_frame_size_bytes
```

- Schema没有区分或保存：
  - loader发送前计算的frame/payload SHA与size；
  - detector对实际收到bytes独立计算的frame/payload SHA与size；
  - detector实际接收的frame count；
  - EOF是否被观察；
  - trailing-byte count。
- `instrumentation-evidence.json`也只有聚合、自报的
  `ipc_envelope_violation_count`，没有per-call双端transcript。
- IPC frame是transient bytes；frame/header/payload在正式outputs或sibling中
  均未保留。Terminal verifier事后只能看到单个hash和zero violation count，
  无法独立重算以下主张：

```text
loader bytes == detector bytes
frame count == 1
EOF observed == true
unused/trailing bytes == 0
```

- 因而一个实现仍可由单一orchestrator值填充三个hash字段并自报zero count，
  而不产生Round 7要求的loader/detector独立证据。这不是纯测试覆盖问题，
  而是final evidence schema缺少所需观测量。

必须修复：
- 为每个call冻结并持久化loader与detector各自独立生成的：
  - header SHA；
  - payload SHA/size；
  - frame SHA/size。
- 增加receiver-side `received_frame_count=1`、
  `eof_observed=true`、`unused_byte_count=0`，并冻结exact类型、排序和等式。
- Terminal verifier必须从instrumentation sibling复核：

```text
loader endpoint == detector endpoint == FeatureCall
received_frame_count == 1
eof_observed == true
unused_byte_count == 0
```

- Hostile minimum需显式包含second frame、missing EOF、parent/inherited send-end
  未关闭、receiver-side hash mutation，而不能只以generic
  `raw-byte smuggling`代替。

### P2-1 Remote old/new transition evidence is not exact for `git push --porcelain`

位置：
- execution plan `:474-510`
- execution plan `:530-566`
- execution plan `:1122-1128`
- execution plan `:1213-1222`
- execution plan `:1743-1745`

问题：
- Exact fetch/push URL、refspec、first-absence、post-consumption和
  post-terminal `ls-remote`结果已经冻结，verifier result也新增了final
  remote identity和observed head。
- 但计划同时要求两次push具有full 40-hex `expected old/new` tuple，并规定
  任何额外output line或不同tuple为terminal failure。
- `git push --porcelain`本身不输出该full tuple：
  - 新建ref的status summary是`[new branch]`，不是40个零；
  - fast-forward summary通常是abbreviated `old..new`，不是两个full SHA。
- 本轮使用隔离的临时local bare remote验证了上述行为；未访问注册remote，
  未触及任何研究数据。
- Plan没有定义：
  - full old/new tuple由哪几个pre/post `ls-remote` observation组合产生；
  - push stdout/stderr允许的exact grammar；
  - 哪些普通`To ...`、status和`Done`行属于允许输出；
  - pre/post observations及derived tuple写入哪个machine-readable artifact。
- Terminal verifier结果只持久化最终remote head。它不能从现有schema区分
  “执行了注册的三次observation和两次push”与“仅在终局观察到同一head”。

必须修复：
- 明确old/new authority是pre/post `ls-remote`的full-SHA组合，不得声称来自
  porcelain summary；或者注册能够提供full expected-old CAS的唯一命令。
- 冻结push stdout/stderr parser或明确其内容不参与authority，仅exit code和
  pre/post observations参与。
- 将三次remote observation、两次derived old/new tuple和exact attempt count
  写入唯一machine-readable ledger/receipt，并由terminal verifier复核。
- Hostile minimum增加pre-existing equal、wrong push URL、extra push URL、
  wrong refspec、third push attempt、wrong pre/post observation和abbreviated
  SHA冒充full tuple。
- Hostile minimum `:1743-1745` 仍只点名GitHub admin；应与 `:512-522`
  的“任何remote writer偏离均在模型外”保持同一actor vocabulary。

### P2-2 IPC array table lacks exact typed and byte-length arithmetic

位置：
- execution plan `:183-190`
- execution plan `:912-924`
- execution plan `:1391-1419`

问题：
- Canonical feature hash使用明确的 `dtype.str`，但IPC nested array row只写：

```text
name,dtype,shape,offset_bytes,length_bytes,value_sha256
```

- 该nested object没有像其他named records一样冻结字段类型和domain，也没有
  明确：
  - `dtype == array.dtype.str`；
  - `shape == list[int]`且各维非负；
  - offset/length为非负int且bool forbidden；
  - `length_bytes == product(shape) * dtype.itemsize`；
  - `value_sha256 == SHA256(payload[offset:offset+length])`；
  - header array row去掉offset/length后的projection必须精确等于canonical
    feature-hash row。
- `validates every table/hash/length`不能替代上述exact arithmetic。不同实现
  可选择`"float64"`或`"<f8"`，或接受shape/length不一致的表，却生成不同
  frame bytes并都自称合规。

必须修复：
- 定义typed `IPCArrayRow` named record及exact dtype authority。
- 冻结shape/itemsize/length/offset/payload-slice hash全部整数等式与
  zero-dimensional/zero-length array语义。
- 增加dtype spelling、shape/length mismatch、offset overflow、
  payload-slice hash和canonical feature-row projection hostile cases。

## Closed Contract Areas

本轮确认已闭合或未发现新回归：
- Revision 8 idea/plan/task SHA identity及pre-execution scope；
- remote authority的name、fetch URL、push URL和ledger ref；
- first remote observation必须empty，pre-existing equal不允许重放；
- threat model明确排除任何持有remote write credential且偏离protocol的actor；
- post-terminal verifier在线观察字段；
- canonical frame由uint64 big-endian header length、header和payload组成；
- payload连续offset、无gap/overlap/trailing及single-frame/EOF规范文本；
- per-call `field_access_sha256`恰为本call的12个完整FieldAccess rows；
- `feature_key_count`在header、loader dict、canonical feature rows和detector
  reconstructed dict之间的四方等式；
- RawOpenEvent phase/operation/path/caller matrix；
- native builtins/io/os open统一规范化为canonical `open`；
- scientific detector顺序、slice identity、source/poison boundary、
  17-output closure、gate/classification precedence、primary/sensitivity
  non-rescue及post-Build-A no-repair lock未发现Revision 8回归。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/1/2/0
```

Revision 8 不可冻结。29-cache、future outcomes和A0 execution lock必须继续
关闭。只有上述findings全部闭合并经下一轮独立review达到`0/0/0/0`，才可进入
implementation/data execution。
