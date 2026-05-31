# Stage 9N Recommendation

- final_classification: `short_collection_to_cross_minimum_only`
- policy_design_status: `blocked`
- top_gap_after: `36` fills
- top_gap_remaining_to_40: `4` fills
- observed_increment_from_120m_sample: `2` fills / 120m
- estimated_time_to_40: `4h-6h`
- estimated_time_to_plus20: `20h+`

The 0530T002 sample added `+108` aggregate Stage 9K fills, but the top Stage 9L clean-fill gap only gained `+2` fills (`34 -> 36`). That gap is still the best clean candidate: it is decision-visible, clean-only, and materially better on quality than the main alternatives.

Why the top gap stayed slow:
- the new sample mostly spread fill mass across other regimes, not the top gap
- the top gap remains low-rate at about `0.012410` fill rate
- alternative regimes are higher-fill-rate, but they are worse on markout and spread capture, so they are not better follow-up targets

Recommendation:
- do a narrow `4h-6h` short collection only to cross the `40` clean-fill minimum
- do not chase the `+20` interpretive target on this line
- do not promote this into policy design or strategy work

Boundary:
- no new data collection in this analysis
- no strategy behavior change
- no candidate enablement
- no guard relaxation
- no parameter search
- no default-on
- no tiny-live
- no promotion
- no replay semantic change
- no connector/core API/schema change
- no Hyperliquid work
