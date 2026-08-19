from examples.hyperliquid.cross_exchange_bbo_lag_surrogate import LAG_GRID
from examples.hyperliquid.cross_exchange_three_session_commonality import (
    PRIMARY_SURROGATE_COUNT,
    RUN_SEED,
    unbiased_index,
)


def test_lag_grid_has_frozen_support_without_zero():
    assert len(LAG_GRID) == 480_002
    assert LAG_GRID[0] == -300_000
    assert LAG_GRID[240_000] == -60_000
    assert LAG_GRID[240_001] == 60_000
    assert LAG_GRID[-1] == 300_000
    assert 0 not in LAG_GRID


def test_all_surrogate_ids_map_inside_grid():
    for surrogate_id in (0, 1, PRIMARY_SURROGATE_COUNT - 1):
        index = unbiased_index(
            ["lag-v1", RUN_SEED, str(surrogate_id), "segment_0001"], len(LAG_GRID)
        )
        assert 0 <= index < len(LAG_GRID)
