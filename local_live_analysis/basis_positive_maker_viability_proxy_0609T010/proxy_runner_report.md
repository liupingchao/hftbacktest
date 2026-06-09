# Basis-Positive Maker-Viability Proxy Runner Report

Task: `0609T010`

## Result

- Final recommendation: `read_only_proxy_evidence_ready_for_qa`
- Source row count: `3545`
- Generated proxy row count: `21270`
- Proxy metric count: `6`

## Boundary

- T010 produces read-only proxy evidence only.
- It does not implement case-library behavior, source-row case catalogs, shadow decisions, executable triggers, trading instructions, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.
- Future labels are output-only offline research labels and are not inputs, filters, triggers, case conditions, shadow-decision fields, live decisions, or deployment criteria.
- Fill probability, exact queue position, exchange post-only reject behavior, cancel-fill race, realized fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, and maker execution viability remain unproven.
