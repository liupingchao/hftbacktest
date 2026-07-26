#!/bin/sh
set -eu

base=/home/admin/hftbacktest-cross-exchange-artifacts
control="$base/0726T068_control_R1"
window_01="$base/0726T068_window_01_R1"
window_02="$base/0726T068_window_02_R1"
window_03="$base/0726T068_window_03_R1"
scan_script=/home/admin/hftbacktest-cross-exchange-control/0726T068/artifact_secret_scan_remote.py
package=/home/admin/0726T068_live_artifacts_R1.tar.gz
bundle_manifest=/home/admin/0726T068_bundle_file_manifest.txt
package_sha=/home/admin/0726T068_live_artifacts_R1.tar.gz.sha256

for root in "$control" "$window_01" "$window_02" "$window_03"; do
  test -d "$root"
done

/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python \
  "$scan_script" \
  --env-file /home/admin/XEMM_rust_latest/.env \
  --output "$control/artifact_secret_scan.json" \
  --root "$control" \
  --root "$window_01" \
  --root "$window_02" \
  --root "$window_03"

jq -e \
  '
    .status == "pass"
    and .identity_key_count > 0
    and .raw_identity_match_count == 0
    and .env_named_file_count == 0
    and .raw_identity_values_written == false
  ' \
  "$control/artifact_secret_scan.json" >/dev/null

for root in "$window_01" "$window_02" "$window_03"; do
  (
    cd "$root"
    sha256sum -c remote_sha256_manifest.txt
  ) >"$root/remote_sha256_recheck.txt"
done

rm -f "$bundle_manifest" "$package" "$package_sha"
find \
  "$control" \
  "$window_01" \
  "$window_02" \
  "$window_03" \
  -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$bundle_manifest"

tar -C /home/admin -czf "$package" \
  hftbacktest-cross-exchange-artifacts/0726T068_control_R1 \
  hftbacktest-cross-exchange-artifacts/0726T068_window_01_R1 \
  hftbacktest-cross-exchange-artifacts/0726T068_window_02_R1 \
  hftbacktest-cross-exchange-artifacts/0726T068_window_03_R1 \
  0726T068_bundle_file_manifest.txt
sha256sum "$package" >"$package_sha"

echo T068_ARTIFACT_PACKAGE_PASS
cat "$package_sha"
