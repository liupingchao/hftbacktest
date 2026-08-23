from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.hyperliquid import trading_runtime_discovery as discovery


SECRET_SENTINELS = {
    "BINANCE_API_KEY": "binance-public-sentinel",
    "BINANCE_API_SECRET": "binance-secret-sentinel",
    "HL_PRIVATE_KEY": "hyperliquid-private-sentinel",
    "HL_WALLET": "0x1111111111111111111111111111111111111111",
}


def _ready_python_probe(path: Path) -> dict[str, object]:
    return {
        "executable": str(path),
        "python_version": "3.13.5",
        "hyperliquid_sdk_version": "0.24.0",
        "hyperliquid_importable": True,
        "exchange_methods": list(
            discovery.REQUIRED_HYPERLIQUID_EXCHANGE_METHODS
        ),
        "info_methods": list(discovery.REQUIRED_HYPERLIQUID_INFO_METHODS),
        "probe_error": "",
    }


def _create_repo(path: Path) -> None:
    (path / "examples/binance_tick_mm").mkdir(parents=True)
    (path / "examples/hyperliquid").mkdir(parents=True)
    (path / discovery.REQUIRED_REPO_FILES["binance_order_runtime"]).write_text(
        "# binance\n",
        encoding="utf-8",
    )
    (
        path / discovery.REQUIRED_REPO_FILES["hyperliquid_order_runtime"]
    ).write_text("# hyperliquid\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def _create_runtime_fixture(tmp_path: Path) -> dict[str, Path]:
    home = tmp_path / "home/admin"
    backing = tmp_path / "srv/crypto-bot/research/home/admin"
    repo = backing / "hftbacktest-cross-exchange"
    env_file = backing / "XEMM_rust_latest/.env"
    venv = home / "0729T003-venv"
    python = venv / "bin/python"
    home.mkdir(parents=True)
    _create_repo(repo)
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in SECRET_SENTINELS.items())
        + "\nSOL_PRIVATE=unrelated-secret-sentinel\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve())
    (home / "hftbacktest-cross-exchange").symlink_to(
        repo, target_is_directory=True
    )
    (home / "XEMM_rust_latest").symlink_to(
        env_file.parent, target_is_directory=True
    )
    return {
        "home": home,
        "backing": backing,
        "repo": repo,
        "env": env_file,
        "venv": venv,
        "python": python,
        "runtime_root": home / "trading",
    }


def test_discovery_follows_home_symlinks_and_never_emits_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _create_runtime_fixture(tmp_path)
    monkeypatch.setattr(discovery, "_probe_python", _ready_python_probe)

    manifest = discovery.discover_runtime(
        home=fixture["home"],
        runtime_root=fixture["runtime_root"],
    )

    assert manifest["lookup_ready"] is True
    assert manifest["execution_runtime_ready"] is True
    assert manifest["selected"]["repo"]["is_symlink"] is True
    assert (
        manifest["selected"]["repo"]["resolved_path"]
        == str(fixture["repo"].resolve())
    )
    assert manifest["selected"]["credentials"]["is_symlink"] is False
    assert all(
        row["ready"]
        for row in manifest["selected"]["credentials"][
            "exchange_status"
        ].values()
    )
    serialized = json.dumps(manifest, sort_keys=True)
    assert all(value not in serialized for value in SECRET_SENTINELS.values())
    assert manifest["boundary_flags"]["credential_values_emitted"] is False


def test_install_creates_private_short_aliases_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _create_runtime_fixture(tmp_path)
    monkeypatch.setattr(discovery, "_probe_python", _ready_python_probe)

    first = discovery.install_aliases(
        home=fixture["home"],
        runtime_root=fixture["runtime_root"],
        repo_target=fixture["repo"],
        env_target=fixture["env"],
        venv_target=fixture["venv"],
        python_target=fixture["python"],
        tool_source=Path(discovery.__file__),
    )
    second = discovery.install_aliases(
        home=fixture["home"],
        runtime_root=fixture["runtime_root"],
        repo_target=fixture["repo"],
        env_target=fixture["env"],
        venv_target=fixture["venv"],
        python_target=fixture["python"],
        tool_source=Path(discovery.__file__),
    )
    third = discovery.install_aliases(
        home=fixture["home"],
        runtime_root=fixture["runtime_root"],
        repo_target=fixture["repo"],
        env_target=fixture["env"],
        venv_target=fixture["venv"],
        python_target=fixture["python"],
        tool_source=fixture["runtime_root"]
        / "bin/trading_runtime_discovery.py",
    )

    assert first["lookup_ready"] is True
    assert second["lookup_ready"] is True
    assert third["lookup_ready"] is True
    assert fixture["runtime_root"].stat().st_mode & 0o777 == 0o700
    for name in ("repo", "credentials.env", ".env", "venv"):
        assert (fixture["runtime_root"] / name).is_symlink()
    assert (fixture["runtime_root"] / "python").is_file()
    assert not (fixture["runtime_root"] / "python").is_symlink()
    assert os.access(fixture["runtime_root"] / "python", os.X_OK)
    assert str(fixture["runtime_root"] / "venv/bin/python") in (
        fixture["runtime_root"] / "python"
    ).read_text(encoding="utf-8")
    assert os.access(fixture["runtime_root"] / "inspect", os.X_OK)
    assert (
        fixture["runtime_root"] / "credentials.env"
    ).resolve() == fixture["env"].resolve()
    manifest_text = (
        fixture["runtime_root"] / "runtime-manifest.json"
    ).read_text(encoding="utf-8")
    assert all(
        value not in manifest_text for value in SECRET_SENTINELS.values()
    )
    assert "SOL_PRIVATE" not in manifest_text
    assert "unrelated-secret-sentinel" not in manifest_text


def test_install_rejects_group_or_world_readable_credential_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _create_runtime_fixture(tmp_path)
    monkeypatch.setattr(discovery, "_probe_python", _ready_python_probe)
    fixture["env"].chmod(0o640)

    with pytest.raises(
        discovery.DiscoveryError,
        match="permissions are not private",
    ):
        discovery.install_aliases(
            home=fixture["home"],
            runtime_root=fixture["runtime_root"],
            repo_target=fixture["repo"],
            env_target=fixture["env"],
            venv_target=fixture["venv"],
            python_target=fixture["python"],
            tool_source=Path(discovery.__file__),
        )


def test_credential_snapshot_reports_names_and_nonempty_status_only(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BINANCE_API_KEY=''\n"
        "BINANCE_API_SECRET=secret-sentinel\n"
        "export HL_PRIVATE_KEY=private-sentinel\n"
        "HL_WALLET=wallet-sentinel\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    snapshot = discovery.credential_snapshot(env_file)

    assert snapshot["exchange_status"]["binance"]["ready"] is False
    assert snapshot["exchange_status"]["hyperliquid"]["ready"] is True
    assert snapshot["credential_values_emitted"] is False
    serialized = json.dumps(snapshot, sort_keys=True)
    for secret in (
        "secret-sentinel",
        "private-sentinel",
        "wallet-sentinel",
    ):
        assert secret not in serialized


def test_discovery_requires_unified_account_and_flatten_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _create_runtime_fixture(tmp_path)
    incomplete_probe = _ready_python_probe(fixture["python"])
    incomplete_probe["exchange_methods"] = [
        method
        for method in incomplete_probe["exchange_methods"]
        if method != "market_close"
    ]
    incomplete_probe["info_methods"] = [
        method
        for method in incomplete_probe["info_methods"]
        if method != "extra_agents"
    ]
    monkeypatch.setattr(
        discovery,
        "_probe_python",
        lambda _: incomplete_probe,
    )

    manifest = discovery.discover_runtime(
        home=fixture["home"],
        runtime_root=fixture["runtime_root"],
    )

    assert manifest["lookup_ready"] is False
    assert manifest["execution_runtime_ready"] is False
    assert "hyperliquid_sdk_surface_incomplete" in (
        manifest["execution_runtime_blockers"]
    )
