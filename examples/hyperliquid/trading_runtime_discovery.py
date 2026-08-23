#!/usr/bin/env python3
"""Discover and install short aliases for private trading runtime inputs.

The tool reads credential key names and non-empty status only. It never emits
credential values, constructs a wallet-backed client, or calls an exchange
endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "trading_runtime_discovery_v1"
DEFAULT_RUNTIME_DIRNAME = "trading"
ENV_ASSIGNMENT_RE = re.compile(
    r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$"
)

CREDENTIAL_REQUIREMENTS = {
    "binance": {
        "api_key": ("BINANCE_API_KEY",),
        "api_secret": ("BINANCE_API_SECRET",),
    },
    "hyperliquid": {
        "private_key": ("HL_PRIVATE_KEY", "HYPERLIQUID_PRIVATE_KEY"),
        "account": ("HL_WALLET", "HYPERLIQUID_ACCOUNT_ADDRESS"),
    },
}

REQUIRED_HYPERLIQUID_EXCHANGE_METHODS = (
    "order",
    "cancel",
    "cancel_by_cloid",
    "schedule_cancel",
    "market_close",
)
REQUIRED_HYPERLIQUID_INFO_METHODS = (
    "open_orders",
    "user_state",
    "spot_user_state",
    "user_fills",
    "user_fills_by_time",
    "query_order_by_oid",
    "query_order_by_cloid",
    "user_role",
    "query_user_abstraction_state",
    "extra_agents",
)
REQUIRED_REPO_FILES = {
    "binance_order_runtime": "examples/binance_tick_mm/live_tick_mm.py",
    "hyperliquid_order_runtime": (
        "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py"
    ),
}

BOUNDARY_FLAGS = {
    "credential_file_read": True,
    "credential_values_emitted": False,
    "credential_values_copied": False,
    "wallet_client_constructed": False,
    "private_endpoint_called": False,
    "account_endpoint_called": False,
    "order_endpoint_called": False,
    "cancel_endpoint_called": False,
}


class DiscoveryError(RuntimeError):
    """Raised when runtime discovery or alias installation must fail closed."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _dedupe_paths(paths: Iterable[Path | None]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _default_backing_home(home: Path) -> Path:
    try:
        relative = home.resolve(strict=False).relative_to("/")
    except ValueError:
        relative = Path(str(home).lstrip("/"))
    return Path("/srv/crypto-bot/research") / relative


def candidate_paths(
    *,
    home: Path,
    runtime_root: Path,
    explicit_repo: Path | None = None,
    explicit_env: Path | None = None,
    explicit_python: Path | None = None,
) -> dict[str, list[Path]]:
    backing_home = _default_backing_home(home)
    venv_candidates = _dedupe_paths(
        [
            runtime_root / "venv",
            home / "0729T003-venv",
            home / ".venvs" / "hyperliquid-sdk-0618T002",
        ]
    )
    return {
        "repo": _dedupe_paths(
            [
                explicit_repo,
                runtime_root / "repo",
                home / "hftbacktest-cross-exchange",
                backing_home / "hftbacktest-cross-exchange",
            ]
        ),
        "env": _dedupe_paths(
            [
                explicit_env,
                runtime_root / "credentials.env",
                runtime_root / ".env",
                home / "XEMM_rust_latest" / ".env",
                backing_home / "XEMM_rust_latest" / ".env",
            ]
        ),
        "python": _dedupe_paths(
            [
                explicit_python,
                runtime_root / "python",
                *(venv / "bin" / "python" for venv in venv_candidates),
                Path(sys.executable),
            ]
        ),
    }


def _path_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "resolved_path": "",
        "kind": "missing",
        "target_mode": "",
        "target_owner_uid": None,
        "target_group_gid": None,
        "resolution_error": "",
    }
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        snapshot["resolution_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot
    snapshot["resolved_path"] = str(resolved)
    if resolved.is_file():
        snapshot["kind"] = "file"
    elif resolved.is_dir():
        snapshot["kind"] = "directory"
    else:
        snapshot["kind"] = "other"
    target_stat = resolved.stat()
    snapshot["target_mode"] = format(stat.S_IMODE(target_stat.st_mode), "03o")
    snapshot["target_owner_uid"] = target_stat.st_uid
    snapshot["target_group_gid"] = target_stat.st_gid
    return snapshot


def _first_existing(
    candidates: list[Path],
    *,
    expected_kind: str,
) -> tuple[Path | None, list[dict[str, Any]]]:
    snapshots = [_path_snapshot(path) for path in candidates]
    for path, snapshot in zip(candidates, snapshots):
        if snapshot["exists"] and snapshot["kind"] == expected_kind:
            return path, snapshots
    return None, snapshots


def _parse_env_key_status(path: Path) -> dict[str, bool]:
    key_status: dict[str, bool] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = ENV_ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        key = match.group(1)
        raw_value = match.group(2).strip()
        if (
            len(raw_value) >= 2
            and raw_value[0] == raw_value[-1]
            and raw_value[0] in {"'", '"'}
        ):
            raw_value = raw_value[1:-1]
        key_status[key] = bool(raw_value.strip())
    return key_status


def credential_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    target_stat = resolved.stat()
    mode = stat.S_IMODE(target_stat.st_mode)
    key_status = _parse_env_key_status(resolved)
    recognized_keys = {
        key
        for requirements in CREDENTIAL_REQUIREMENTS.values()
        for alternatives in requirements.values()
        for key in alternatives
    }
    exchange_status: dict[str, Any] = {}
    for exchange, requirements in CREDENTIAL_REQUIREMENTS.items():
        checks: dict[str, Any] = {}
        for label, alternatives in requirements.items():
            present_keys = sorted(key for key in alternatives if key in key_status)
            nonempty_keys = sorted(
                key for key in alternatives if key_status.get(key, False)
            )
            checks[label] = {
                "accepted_key_names": list(alternatives),
                "present_key_names": present_keys,
                "nonempty_key_names": nonempty_keys,
                "ready": bool(nonempty_keys),
            }
        exchange_status[exchange] = {
            "ready": all(check["ready"] for check in checks.values()),
            "requirements": checks,
        }
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        "target_mode": format(mode, "03o"),
        "target_owner_uid": target_stat.st_uid,
        "target_group_gid": target_stat.st_gid,
        "permission_secure": mode & 0o077 == 0,
        "recognized_key_names": sorted(
            key for key in key_status if key in recognized_keys
        ),
        "recognized_nonempty_key_names": sorted(
            key
            for key, is_nonempty in key_status.items()
            if key in recognized_keys and is_nonempty
        ),
        "unrecognized_key_count": len(set(key_status) - recognized_keys),
        "exchange_status": exchange_status,
        "credential_values_emitted": False,
    }


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def repo_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    branch = _run_git(resolved, "branch", "--show-current")
    commit = _run_git(resolved, "rev-parse", "HEAD")
    status_lines = [
        line
        for line in _run_git(resolved, "status", "--porcelain").splitlines()
        if line
    ]
    source_presence = {
        label: (resolved / relative_path).is_file()
        for label, relative_path in REQUIRED_REPO_FILES.items()
    }
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        "is_git_checkout": bool(commit),
        "branch": branch,
        "commit": commit,
        "dirty_count": len(status_lines),
        "working_tree_clean": bool(commit) and not status_lines,
        "source_presence": source_presence,
        "order_runtime_sources_present": all(source_presence.values()),
    }


def _probe_python(path: Path) -> dict[str, Any]:
    probe = """
import importlib.metadata as metadata
import json
import sys

result = {
    "executable": sys.executable,
    "python_version": sys.version.split()[0],
    "hyperliquid_sdk_version": "",
    "hyperliquid_importable": False,
    "exchange_methods": [],
    "info_methods": [],
    "probe_error": "",
}
try:
    result["hyperliquid_sdk_version"] = metadata.version("hyperliquid-python-sdk")
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    result["hyperliquid_importable"] = True
    result["exchange_methods"] = sorted(
        name for name in %r if callable(getattr(Exchange, name, None))
    )
    result["info_methods"] = sorted(
        name for name in %r if callable(getattr(Info, name, None))
    )
except Exception as exc:
    result["probe_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
""" % (
        REQUIRED_HYPERLIQUID_EXCHANGE_METHODS,
        REQUIRED_HYPERLIQUID_INFO_METHODS,
    )
    completed = subprocess.run(
        [str(path), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        return {
            "executable": str(path),
            "python_version": "",
            "hyperliquid_sdk_version": "",
            "hyperliquid_importable": False,
            "exchange_methods": [],
            "info_methods": [],
            "probe_error": (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"python_probe_exit_{completed.returncode}"
            ),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "executable": str(path),
            "python_version": "",
            "hyperliquid_sdk_version": "",
            "hyperliquid_importable": False,
            "exchange_methods": [],
            "info_methods": [],
            "probe_error": f"invalid_probe_json: {exc}",
        }


def python_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    probe = _probe_python(resolved)
    exchange_methods = set(probe.get("exchange_methods", []))
    info_methods = set(probe.get("info_methods", []))
    surface_ready = (
        set(REQUIRED_HYPERLIQUID_EXCHANGE_METHODS) <= exchange_methods
        and set(REQUIRED_HYPERLIQUID_INFO_METHODS) <= info_methods
    )
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        **probe,
        "hyperliquid_order_cancel_surface_ready": surface_ready,
    }


def discover_runtime(
    *,
    home: Path,
    runtime_root: Path,
    explicit_repo: Path | None = None,
    explicit_env: Path | None = None,
    explicit_python: Path | None = None,
) -> dict[str, Any]:
    candidates = candidate_paths(
        home=home,
        runtime_root=runtime_root,
        explicit_repo=explicit_repo,
        explicit_env=explicit_env,
        explicit_python=explicit_python,
    )
    selected_repo, repo_candidates = _first_existing(
        candidates["repo"], expected_kind="directory"
    )
    selected_env, env_candidates = _first_existing(
        candidates["env"], expected_kind="file"
    )
    selected_python, python_candidates = _first_existing(
        candidates["python"], expected_kind="file"
    )
    repo = repo_snapshot(selected_repo) if selected_repo else None
    credentials = credential_snapshot(selected_env) if selected_env else None
    python = python_snapshot(selected_python) if selected_python else None
    credential_groups_ready = bool(
        credentials
        and credentials["permission_secure"]
        and all(
            row["ready"]
            for row in credentials["exchange_status"].values()
        )
    )
    lookup_ready = bool(
        repo
        and repo["order_runtime_sources_present"]
        and credentials
        and credential_groups_ready
        and python
        and python["hyperliquid_order_cancel_surface_ready"]
    )
    execution_runtime_ready = bool(
        lookup_ready and repo and repo["working_tree_clean"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "home": str(home),
        "runtime_root": str(runtime_root),
        "discovery_status": "complete" if lookup_ready else "incomplete",
        "lookup_ready": lookup_ready,
        "execution_runtime_ready": execution_runtime_ready,
        "execution_runtime_blockers": (
            []
            if execution_runtime_ready
            else [
                reason
                for reason, blocked in (
                    ("repo_missing_or_order_sources_incomplete", not repo or not repo["order_runtime_sources_present"]),
                    ("credential_groups_missing_or_permissions_insecure", not credential_groups_ready),
                    ("hyperliquid_sdk_surface_incomplete", not python or not python["hyperliquid_order_cancel_surface_ready"]),
                    ("repo_working_tree_not_clean", bool(repo) and not repo["working_tree_clean"]),
                )
                if blocked
            ]
        ),
        "selected": {
            "repo": repo,
            "credentials": credentials,
            "python": python,
        },
        "candidates": {
            "repo": repo_candidates,
            "env": env_candidates,
            "python": python_candidates,
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }


def _atomic_symlink(link: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        try:
            if link.resolve(strict=True) == target:
                return
        except OSError:
            pass
    elif link.exists():
        raise DiscoveryError(f"refusing to replace non-symlink path: {link}")
    temporary = link.parent / f".{link.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target, target_is_directory=target.is_dir())
    os.replace(temporary, link)


def _write_private_text(path: Path, content: str, *, executable: bool = False) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o700 if executable else 0o600)
    os.replace(temporary, path)


def install_aliases(
    *,
    home: Path,
    runtime_root: Path,
    repo_target: Path,
    env_target: Path,
    venv_target: Path,
    python_target: Path,
    tool_source: Path,
) -> dict[str, Any]:
    if runtime_root.is_symlink():
        raise DiscoveryError(f"runtime root must not be a symlink: {runtime_root}")
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_root.chmod(0o700)
    bin_dir = runtime_root / "bin"
    bin_dir.mkdir(mode=0o700, exist_ok=True)
    bin_dir.chmod(0o700)

    repo_resolved = repo_target.resolve(strict=True)
    env_resolved = env_target.resolve(strict=True)
    venv_resolved = venv_target.resolve(strict=True)
    python_resolved = python_target.resolve(strict=True)
    if not repo_resolved.is_dir():
        raise DiscoveryError(f"repo target is not a directory: {repo_target}")
    if not env_resolved.is_file():
        raise DiscoveryError(f"env target is not a file: {env_target}")
    if not venv_resolved.is_dir():
        raise DiscoveryError(f"venv target is not a directory: {venv_target}")
    if not python_resolved.is_file() or not os.access(python_resolved, os.X_OK):
        raise DiscoveryError(f"python target is not executable: {python_target}")
    expected_venv_python = (venv_resolved / "bin" / "python").resolve(
        strict=True
    )
    if python_resolved != expected_venv_python:
        raise DiscoveryError(
            "python target does not belong to the selected venv: "
            f"{python_target}"
        )

    credentials = credential_snapshot(env_resolved)
    if not credentials["permission_secure"]:
        raise DiscoveryError(
            f"credential target permissions are not private: {env_resolved}"
        )
    missing_exchanges = sorted(
        exchange
        for exchange, row in credentials["exchange_status"].items()
        if not row["ready"]
    )
    if missing_exchanges:
        raise DiscoveryError(
            "credential target is missing required key groups: "
            + ",".join(missing_exchanges)
        )

    _atomic_symlink(runtime_root / "repo", repo_resolved)
    _atomic_symlink(runtime_root / "credentials.env", env_resolved)
    _atomic_symlink(runtime_root / ".env", env_resolved)
    _atomic_symlink(runtime_root / "venv", venv_resolved)
    python_alias = runtime_root / "python"
    if python_alias.is_symlink():
        python_alias.unlink()
    elif python_alias.exists() and not python_alias.is_file():
        raise DiscoveryError(
            f"refusing to replace non-file Python alias: {python_alias}"
        )
    _write_private_text(
        python_alias,
        "\n".join(
            [
                "#!/bin/sh",
                f'exec "{runtime_root / "venv" / "bin" / "python"}" "$@"',
                "",
            ]
        ),
        executable=True,
    )

    installed_tool = bin_dir / "trading_runtime_discovery.py"
    if tool_source.resolve(strict=True) != installed_tool.resolve(strict=False):
        shutil.copyfile(tool_source.resolve(strict=True), installed_tool)
    installed_tool.chmod(0o700)
    inspect_script = runtime_root / "inspect"
    _write_private_text(
        inspect_script,
        "\n".join(
            [
                "#!/bin/sh",
                f'exec "{runtime_root / "python"}" '
                f'"{installed_tool}" '
                f'inspect --home "{home}" --runtime-root "{runtime_root}" "$@"',
                "",
            ]
        ),
        executable=True,
    )
    readme = runtime_root / "README.md"
    _write_private_text(
        readme,
        "\n".join(
            [
                "# Trading Runtime",
                "",
                "- Inspect without exposing values: `~/trading/inspect`",
                "- JSON inspection: `~/trading/inspect --json`",
                "- Checkout: `cd ~/trading/repo`",
                "- Python: `~/trading/python`",
                "- Credential input: `~/trading/credentials.env`",
                "",
                "The credential aliases point to one existing mode-600 source.",
                "Do not print, copy, source in debug mode, or commit its values.",
                "",
            ]
        ),
    )
    manifest = discover_runtime(home=home, runtime_root=runtime_root)
    manifest_path = runtime_root / "runtime-manifest.json"
    _write_private_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_private_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _print_human(payload: dict[str, Any]) -> None:
    selected = payload["selected"]
    print(f"discovery_status={payload['discovery_status']}")
    print(f"lookup_ready={str(payload['lookup_ready']).lower()}")
    print(
        "execution_runtime_ready="
        f"{str(payload['execution_runtime_ready']).lower()}"
    )
    for label in ("repo", "credentials", "python"):
        row = selected[label]
        if row is None:
            print(f"{label}=missing")
        else:
            print(
                f"{label}={row['path']} -> {row['resolved_path']}"
            )
    credentials = selected["credentials"]
    if credentials:
        for exchange, row in credentials["exchange_status"].items():
            print(
                f"{exchange}_credentials_ready="
                f"{str(row['ready']).lower()}"
            )
    python = selected["python"]
    if python:
        print(
            "hyperliquid_sdk="
            f"{python['hyperliquid_sdk_version'] or 'missing'}"
        )
    print("credential_values_emitted=false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="discover repo, credential and Python candidates without endpoints",
    )
    inspect_parser.add_argument("--home", type=Path, default=Path.home())
    inspect_parser.add_argument("--runtime-root", type=Path)
    inspect_parser.add_argument("--repo", type=Path)
    inspect_parser.add_argument("--env-file", type=Path)
    inspect_parser.add_argument("--python", type=Path)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.add_argument("--json", action="store_true")

    install_parser = subparsers.add_parser(
        "install",
        help="install private short aliases without copying credentials",
    )
    install_parser.add_argument("--home", type=Path, default=Path.home())
    install_parser.add_argument("--runtime-root", type=Path)
    install_parser.add_argument("--repo-target", type=Path, required=True)
    install_parser.add_argument("--env-target", type=Path, required=True)
    install_parser.add_argument("--venv-target", type=Path, required=True)
    install_parser.add_argument("--python-target", type=Path, required=True)
    install_parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    home = args.home.expanduser()
    runtime_root = (
        args.runtime_root.expanduser()
        if args.runtime_root
        else home / DEFAULT_RUNTIME_DIRNAME
    )
    try:
        if args.command == "inspect":
            payload = discover_runtime(
                home=home,
                runtime_root=runtime_root,
                explicit_repo=args.repo,
                explicit_env=args.env_file,
                explicit_python=args.python,
            )
            if args.output:
                _write_json(args.output.expanduser(), payload)
        else:
            payload = install_aliases(
                home=home,
                runtime_root=runtime_root,
                repo_target=args.repo_target.expanduser(),
                env_target=args.env_target.expanduser(),
                venv_target=args.venv_target.expanduser(),
                python_target=args.python_target.expanduser(),
                tool_source=Path(__file__),
            )
    except (DiscoveryError, OSError, subprocess.SubprocessError) as exc:
        print(f"trading_runtime_discovery_error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0 if payload["discovery_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
