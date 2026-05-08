"""Scenario harnesses for integration and regression tests.

Each sub-module exposes a single ``setup() -> dict`` factory that returns a
fully populated scenario bundle:

    {
        "name":             str,
        "description":      str,
        "ticks":            list[dict],       # Upstox V3 WS tick dicts
        "warmup_bars":      dict[str, pd.DataFrame],
        "option_chains":    dict[str, pd.DataFrame],
        "news":             list[dict],       # article dicts for make_rss_xml
        "macro_state":      dict,             # spot scalars / regime flags
        "expected_outcome": dict,             # signal_count_min, orders_placed_min, rejection_reason
    }
"""
