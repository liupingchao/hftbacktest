# Live alignment summary: 5-19-day-control-30min_btcusdt_1779182038

## Inputs

- Requested run ID: `5-19-day-control-30min`
- Live audit run ID: `5-19-day-control-30min_btcusdt_1779182038`
- Audit CSV: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/audit_live_5-19-day-control-30min.csv`
- Live rows: `122124`
- First ts_local: `1779182039784265578`
- Last ts_local: `1779184005966369364`
- Start/end day UTC: `2026-05-19` / `2026-05-19`
- Live raw manifest: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/out/live_raw/btcusdt/manifest_2026-05-19_to_2026-05-19.json`
- Initial position source: `rest_position`
- Initial position: `0.0`

## Latency Model

- mode: `observed`
- rows: `95764.0`
- entry_mean_ms: `244.14014207283526`
- entry_p50_ms: `2.915721`
- entry_p90_ms: `534.368808`
- entry_p99_ms: `4350.246115`
- entry_max_ms: `37938.610179`
- resp_mean_ms: `18.28988338376634`
- resp_p50_ms: `3.4968744999999997`
- resp_p90_ms: `33.38416189999997`
- resp_p99_ms: `292.709568`
- resp_max_ms: `2265.3315`
- entry_gt_5ms_ratio: `0.44932333653565015`

## Backtest Alignment

### normal

- Audit CSV: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/out/backtest_normal/audit_bt_normal.csv`
- Rows: `153454`
- Audit rows written: `153462`
- Alignment report: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/alignment_report_normal.json`
- Slice ts_local start/end: `1779182039784265578` / `1779184005966369364`
- Audit replay consumed/scheduled: `0` / `0`
- Action match rate: `0.9289710507468264`
- Reject reason match rate: `0.6233794542302862`
- BT/live latency drop: `0.10860583627666923` / `0.1485764108738751`
- BT/live API drop: `0.0` / `0.16107368617722465`
- MAE: `{'fair': 61.09508587205804, 'reservation': 61.09508618244851, 'half_spread': 8.977815793832375e-06, 'position': 0.0019734588596755274, 'inventory_score': 0.632915027523761, 'spread_bps': 0.0061485280667223936, 'vol_bps': 0.055571578541748545}`

### audit_replay

- Audit CSV: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/out/backtest_audit_replay/audit_bt_audit_replay.csv`
- Rows: `96340`
- Audit rows written: `6026122`
- Alignment report: `/home/molly/project/hftbacktest/local_live_analysis/5-19-day-control-30min/alignment_report_audit_replay.json`
- Slice ts_local start/end: `1779182036784187029` / `1779184005966369364`
- Decision first ts_local: `1779182039784265578`
- First feed ts_local: `1779182039784187029`
- Audit replay prewarm ms: `3000.0`
- Strict lag gate passed/breaches: `True` / `0`
- Audit replay consumed/scheduled: `96340` / `96341`
- Action match rate: `1.0`
- Reject reason match rate: `1.0`
- BT/live latency drop: `0.14857795308283164` / `0.1485764108738751`
- BT/live API drop: `0.16107535810670542` / `0.16107368617722465`
- MAE: `{'fair': 0.0, 'reservation': 0.0, 'half_spread': 0.0, 'position': 0.0, 'inventory_score': 0.0, 'spread_bps': 0.0, 'vol_bps': 0.0}`

## Archive

- Archive: `/home/molly/project/hftbacktest/local_live_analysis/archive/5-19-day-control-30min.tar.gz`
