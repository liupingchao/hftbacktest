from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable


MANIFEST_PATH = Path(
    "baselines/skhynix_fixed_epoch_suppression_v1/"
    "baseline_manifest.json"
)


class BaselineError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256_bytes(encoded)


def git_output(repo_root: Path, *args: str, binary: bool = False) -> Any:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=not binary,
    )


def load_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root / MANIFEST_PATH
    return json.loads(path.read_text(encoding="ascii"))


def function_ast_hashes(
    source: str, names: Iterable[str]
) -> dict[str, str]:
    requested = set(names)
    result: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in requested:
            continue
        dumped = ast.dump(
            node, annotate_fields=True, include_attributes=False
        ).encode("ascii")
        result[node.name] = sha256_bytes(dumped)
    return result


def literal_assignments(
    source: str, names: Iterable[str]
) -> dict[str, Any]:
    requested = set(names)
    result: dict[str, Any] = {}

    def evaluate(node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in result:
            return result[node.id]
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
        raise BaselineError("unsupported_frozen_constant_expression")

    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in requested:
            result[target.id] = evaluate(node.value)
    return result


def require(condition: bool, code: str) -> None:
    if not condition:
        raise BaselineError(code)


def verify_authority(
    repo_root: Path,
    manifest: dict[str, Any],
    *,
    check_working_tree: bool,
) -> dict[str, Any]:
    authority = manifest["git_authority"]["accepted_workflow_commit"]
    tracked = manifest["tracked_authority_files"]
    for path_text, expected in tracked.items():
        authority_bytes = git_output(
            repo_root,
            "show",
            f"{authority}:{path_text}",
            binary=True,
        )
        require(
            sha256_bytes(authority_bytes) == expected["sha256"],
            f"authority_sha256_mismatch:{path_text}",
        )
        blob_oid = git_output(
            repo_root, "rev-parse", f"{authority}:{path_text}"
        ).strip()
        require(
            blob_oid == expected["blob_oid"],
            f"authority_blob_oid_mismatch:{path_text}",
        )
        if check_working_tree:
            working_path = repo_root / path_text
            require(
                working_path.is_file(),
                f"working_tree_file_missing:{path_text}",
            )
            require(
                sha256_file(working_path) == expected["sha256"],
                f"working_tree_sha256_mismatch:{path_text}",
            )

    key_order = (
        "frozen_plan_commit",
        "implementation_commit",
        "hostile_closure_commit",
        "qa_acceptance_commit",
        "accepted_workflow_commit",
    )
    commits = manifest["git_authority"]
    for earlier, later in zip(key_order, key_order[1:]):
        status = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                commits[earlier],
                commits[later],
            ],
            cwd=repo_root,
            check=False,
        ).returncode
        require(status == 0, f"authority_ancestry_mismatch:{earlier}:{later}")

    runner_path = manifest["runner_path"]
    source = git_output(
        repo_root, "show", f"{authority}:{runner_path}"
    )
    expected_ast = manifest["frozen_callable_ast_sha256"]
    require(
        function_ast_hashes(source, expected_ast) == expected_ast,
        "authority_callable_ast_mismatch",
    )
    expected_contract = {
        key: value
        for key, value in manifest["frozen_contract"].items()
        if key.isupper()
    }
    require(
        literal_assignments(source, expected_contract) == expected_contract,
        "authority_fixed_contract_mismatch",
    )
    return {
        "accepted_workflow_commit": authority,
        "authority_file_count": len(tracked),
        "callable_count": len(expected_ast),
        "working_tree_checked": check_working_tree,
    }


def artifact_rows_from_directory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "cache":
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def validate_artifact_rows(
    rows: list[dict[str, Any]],
    run_manifest_bytes: bytes,
    expected: dict[str, Any],
) -> dict[str, Any]:
    require(
        len(rows) == expected["expected_non_cache_artifact_count"],
        "non_cache_artifact_count_mismatch",
    )
    require(
        canonical_sha(rows) == expected["non_cache_tree_sha256"],
        "non_cache_tree_sha256_mismatch",
    )
    by_path = {row["path"]: row for row in rows}
    require(len(by_path) == len(rows), "duplicate_artifact_path")
    require(
        by_path["run_manifest.json"]["sha256"]
        == expected["run_manifest_sha256"],
        "run_manifest_sha256_mismatch",
    )
    require(
        by_path["reports/A_minus1_summary.json"]["sha256"]
        == expected["summary_sha256"],
        "summary_sha256_mismatch",
    )
    run_manifest = json.loads(run_manifest_bytes)
    listed = {row["path"]: row for row in run_manifest["artifacts"]}
    require(
        run_manifest["artifact_count"] == len(rows) - 1,
        "run_manifest_artifact_count_mismatch",
    )
    require(
        set(listed) == set(by_path) - {"run_manifest.json"},
        "run_manifest_path_set_mismatch",
    )
    for path_text, listed_row in listed.items():
        actual = by_path[path_text]
        require(
            listed_row["sha256"] == actual["sha256"]
            and listed_row["size_bytes"] == actual["size_bytes"],
            f"run_manifest_entry_mismatch:{path_text}",
        )
    return {
        "artifact_count": len(rows),
        "tree_sha256": canonical_sha(rows),
    }


def verify_output_root(
    output_root: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    rows = artifact_rows_from_directory(output_root)
    run_manifest_path = output_root / "run_manifest.json"
    require(run_manifest_path.is_file(), "run_manifest_missing")
    result = validate_artifact_rows(
        rows, run_manifest_path.read_bytes(), expected
    )
    summary = json.loads(
        (output_root / "reports/A_minus1_summary.json").read_text(
            encoding="ascii"
        )
    )
    require(
        summary["classification"] == expected["classification"],
        "classification_mismatch",
    )
    require(
        summary["common_candidate_count"] == expected["candidate_count"],
        "candidate_count_mismatch",
    )
    require(
        summary["integrity"]["slice_invariance_mismatches"]
        == expected["slice_invariance_mismatches"],
        "slice_invariance_mismatch",
    )
    return result


def verify_snapshot(
    repo_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    snapshot_files = manifest["tracked_snapshot_files"]
    for path_text, expected in snapshot_files.items():
        path = repo_root / path_text
        require(path.is_file(), f"snapshot_file_missing:{path_text}")
        require(
            sha256_file(path) == expected["sha256"],
            f"snapshot_sha256_mismatch:{path_text}",
        )

    archive_path = (
        repo_root
        / "baselines/skhynix_fixed_epoch_suppression_v1/"
        "evidence_noncache.tar.gz"
    )
    rows = []
    run_manifest_bytes = b""
    with tarfile.open(archive_path, "r:gz") as handle:
        for member in sorted(handle.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            require(
                not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts,
                f"unsafe_archive_member:{member.name}",
            )
            extracted = handle.extractfile(member)
            require(extracted is not None, f"archive_member_missing:{member.name}")
            content = extracted.read()
            rows.append(
                {
                    "path": member.name,
                    "sha256": sha256_bytes(content),
                    "size_bytes": len(content),
                }
            )
            if member.name == "run_manifest.json":
                run_manifest_bytes = content
    require(bool(run_manifest_bytes), "archive_run_manifest_missing")
    result = validate_artifact_rows(
        rows, run_manifest_bytes, manifest["formal_evidence"]
    )
    attestation = json.loads(
        (
            repo_root
            / "baselines/skhynix_fixed_epoch_suppression_v1/"
            "poison-attestation.json"
        ).read_text(encoding="ascii")
    )
    require(
        attestation["changed_unconsumed_field_instance_count"] == 435
        and attestation["consumed_field_mismatch_count"] == 0,
        "poison_attestation_semantics_mismatch",
    )
    return result


def compare_output_roots(
    roots: dict[str, Path], expected: dict[str, Any]
) -> dict[str, Any]:
    trees = {}
    for label, root in roots.items():
        verify_output_root(root, expected)
        trees[label] = {
            row["path"]: row["sha256"]
            for row in artifact_rows_from_directory(root)
        }
    labels = list(trees)
    reference = trees[labels[0]]
    for label in labels[1:]:
        require(
            trees[label] == reference,
            f"triad_output_mismatch:{labels[0]}:{label}",
        )
    return {"labels": labels, "difference_count": 0}


def verify_tags(repo_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    authority = manifest["git_authority"]["accepted_workflow_commit"]
    tags = manifest["recovery_tags"]
    authority_target = git_output(
        repo_root, "rev-parse", f"{tags['suppression_authority']}^{{}}"
    ).strip()
    require(
        authority_target == authority,
        "suppression_authority_tag_target_mismatch",
    )
    kit_target = git_output(
        repo_root, "rev-parse", f"{tags['research_kit']}^{{}}"
    ).strip()
    status = subprocess.run(
        ["git", "merge-base", "--is-ancestor", authority, kit_target],
        cwd=repo_root,
        check=False,
    ).returncode
    require(status == 0, "research_kit_missing_authority_ancestor")
    git_output(
        repo_root, "cat-file", "-e", f"{kit_target}:{MANIFEST_PATH}"
    )
    return {
        "suppression_authority": authority_target,
        "research_kit": kit_target,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-working-tree", action="store_true")
    parser.add_argument("--require-tags", action="store_true")
    parser.add_argument("--canonical-output", type=Path)
    parser.add_argument("--build-b-output", type=Path)
    parser.add_argument("--poison-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    try:
        manifest = load_manifest(repo_root)
        result: dict[str, Any] = {
            "baseline_id": manifest["baseline_id"],
            "authority": verify_authority(
                repo_root,
                manifest,
                check_working_tree=args.check_working_tree,
            ),
            "snapshot": verify_snapshot(repo_root, manifest),
        }
        output_args = (
            args.canonical_output,
            args.build_b_output,
            args.poison_output,
        )
        if any(output_args):
            require(all(output_args), "triad_output_arguments_incomplete")
            result["triad"] = compare_output_roots(
                {
                    "canonical": args.canonical_output.resolve(),
                    "build_b": args.build_b_output.resolve(),
                    "poison": args.poison_output.resolve(),
                },
                manifest["formal_evidence"],
            )
        if args.require_tags:
            result["tags"] = verify_tags(repo_root, manifest)
        result["status"] = "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        BaselineError,
        FileNotFoundError,
        KeyError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
