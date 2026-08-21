# Research Package Task Template

执行线程：
- 业务线程-python/research

任务ID：
- `<MMDDTxxx>`

标题：
- `<TITLE>`

状态：
- 待执行

task_type：
- `research_package` / `research_package_infrastructure`

produces_research_package：
- `true`

kernel pin：
- `mode=accepted` or `mode=bootstrap_candidate`
- Declare every field required by the frozen surface-matrix schema.

Canonical matrix：
- `.workflow/contracts/<TASK_ID>-surface-matrix.json`

Surface Matrix：

| Surface | Authoritative source | Decision/as-of time | Exact fields/keys | Rebuild oracle | Negative mutation | Durable evidence | Identity layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `<surface_id>` | `<exact source and identity>` | `<exact rule>` | `<exact universe>` | `<entrypoint>` | `<mutation and stable code>` | `<path/SHA or frozen reason>` | `R/C/E` |

七条 exit criteria：
- `EC1` Every load-bearing surface has an authoritative source.
- `EC2` Every field is source-derived or explicitly unavailable.
- `EC3` Every evidence object has an exact key/type/value universe.
- `EC4` Every filesystem entry belongs to the exact identity universe.
- `EC5` Every unchanged claim has durable prior evidence.
- `EC6` Every surface has a stable-code negative mutation.
- `EC7` Hostile preflight passes before the first full build/admission.

规则：
- The canonical JSON matrix is the machine source of truth.
- The Markdown table must list the same surface IDs in the same order.
- Empty cells, `TBD`, free-text unavailable reasons and `same as source` are
  forbidden.
- A bootstrap candidate may be used only by
  `task_type=research_package_infrastructure`.
- Business and QA must not promote the accepted-version registry.
