#!/bin/sh
set -eu

repo=/home/admin/hftbacktest-cross-exchange-0726T068R1
control=/home/admin/hftbacktest-cross-exchange-control/0726T068
artifacts=/home/admin/hftbacktest-cross-exchange-artifacts/0726T068_control_R1
expected_source=a0bc92898ecea43cbdc4219efc1acbcd69580969
expected_account=59153858d04cb15ef51660a62b2598b785468306e4d8ab36bdbbfa18ac134c6d
expected_signer=9dd9fcfdb4e3b3e077ba25b41ccc3514d824b952a570fd5a98c3188ec72de422

test "$(cat "$repo/source_commit.txt")" = "$expected_source"
test "$(systemctl is-active xemm.service || true)" = inactive
runuser -u admin -- flock -n /tmp/hftbacktest_live_test.lock -c true

live_processes="$(
  pgrep -af \
    'cross_exchange_live_remote_orchestrator.py|hyperliquid_tiny_live_m2_public_watcher.py|remote_three_window_controller.py' \
    || true
)"
test -z "$live_processes"

runuser -u admin -- \
  /home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python \
  "$control/account_identity_guard.py" \
  --repo "$repo" \
  --env-file /home/admin/XEMM_rust_latest/.env \
  --phase independent_terminal_guard \
  --output "$artifacts/independent_terminal_account_guard.json" \
  --expected-source-commit "$expected_source" \
  --expected-account-scope-sha256 "$expected_account" \
  --expected-signer-sha256 "$expected_signer" \
  --max-abs-btc-position 0.01 \
  --require-open-orders-empty

jq -e \
  --arg account "$expected_account" \
  --arg signer "$expected_signer" \
  '
    .status == "pass"
    and .open_orders_empty == true
    and .open_orders_count == 0
    and .btc_position == 0
    and .position_within_cap == true
    and .raw_credentials_written == false
    and .account_scope_sha256 == $account
    and .signer_sha256 == $signer
  ' \
  "$artifacts/independent_terminal_account_guard.json" >/dev/null

sha256sum "$artifacts/independent_terminal_account_guard.json"
echo T068_INDEPENDENT_TERMINAL_GUARD_PASS
