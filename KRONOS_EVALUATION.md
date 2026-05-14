# Kronos Evaluation — Foundation Model for Financial Time-Series

**Repo:** https://github.com/shiyu-coder/Kronos (default branch `master`)
**License:** MIT (commercial use OK)
**Branch for this eval:** `feature/kronos-evaluation`
**Date:** 2026-05-13

---

## What it is

Open-source **decoder-only autoregressive Transformer** trained on K-line (OHLCV) candle sequences from 45+ global exchanges. Treats financial bars as a "language" — tokenizes continuous OHLCV into discrete tokens via a custom hierarchical tokenizer, then forecasts next-N tokens autoregressively.

**Output:** pandas DataFrame with predicted `open/high/low/close/volume/amount` columns for the next N timesteps.

## Models available (HuggingFace, no auth)

| Model | Params | Context | Notes |
|---|---|---|---|
| `NeoQuasar/Kronos-mini` | 4.1M | 2048 | Tiny, fast, weaker |
| `NeoQuasar/Kronos-small` | 24.7M | 512 | Default in examples |
| `NeoQuasar/Kronos-base` | 102.3M | 512 | Best public |
| Kronos-large | 499M | – | Proprietary, not released |

Plus a tokenizer model (e.g. `NeoQuasar/Kronos-Tokenizer-base`).

## Reference inference shape

- **Input:** 400 historical bars (OHLCV+amount), single instrument
- **Output:** 120-bar forecast
- **Sampling:** temperature=1.0, top_p=0.9
- **Compute:** CPU works for `small`/`base`. GPU faster but not required for single-symbol inference.

---

## Pros for our bot

| Pro | Why it matters |
|---|---|
| **MIT license** | No legal/commercial blockers |
| **Pre-trained foundation model** | Skip training cost — load weights, predict |
| **Probabilistic forecasts** | `T` and `top_p` give us distribution-like predictions, not just point estimates |
| **OHLCV input/output** | Matches our existing `bars` table directly |
| **No external API** | Local inference — zero per-call cost, no rate limits, no privacy concerns |
| **Multi-step forecasts** | 120-bar horizon (e.g. next 10h on 5m bars) — useful for swing entry timing |
| **Plays well with our Confluence Engine** | Becomes a new `ConfluenceDimension` (e.g. `MODEL_FORECAST`) |

## Cons / risks

| Con | Severity |
|---|---|
| **Indian markets NOT explicitly in training data** | HIGH — README says "45+ global exchanges" but examples + finetune scripts focus on Chinese A-share. NSE/BSE coverage unconfirmed. May give garbage out-of-distribution predictions on NIFTY/BANKNIFTY. |
| **Lookahead/leakage caveats not documented** | HIGH — backtester (`finetune/qlib_test.py`) doesn't explicitly safeguard train/test split. Public benchmark numbers absent. |
| **No published benchmark vs LSTM/baseline** | MEDIUM — no proof it beats a simple model |
| **512-token context cap (`small`/`base`)** | MEDIUM — restricts to ~500 prior bars. Fine for intraday but limits multi-day macro context |
| **Latency unmeasured** | MEDIUM — likely 100-1000ms per inference on CPU. OK for 5m+ bars, too slow for tick-level |
| **Volume/amount in INR scale unclear** | LOW — input scaling instructions thin; could miscalibrate on Indian rupee values |
| **Author is research-grade, not enterprise** | LOW — single-author repo, docs say "not production-ready quantitative trading system" |

---

## Verdict

**Useful — but only as an additional confluence signal, NOT as standalone signal source.**

3 reasons to add it:
1. **Free directional prior**: feed predicted close (15m / 30m ahead) as one input into `ConfluenceEngine` — cheap edge if it works
2. **Volatility forecast**: predicted high-low range = useful for SL/target sizing input to KellySizer
3. **Regime hint**: predicted vs actual divergence = regime-change indicator

3 reasons to NOT make it primary:
1. India coverage unverified (HIGH risk of out-of-distribution failures on Indian indices)
2. No published edge benchmarks
3. Doc says "not production ready"

---

## Recommended integration plan (Phase G — IF we proceed)

### G.1 — Model wrapper (1 day)

`src/research/kronos_predictor.py`:
```python
class KronosForecaster:
    def __init__(self, model_size="small", device="cpu"): ...
    def load_warmup_bars(self, instrument_key, lookback=400) -> pd.DataFrame: ...
    def predict(self, bars: pd.DataFrame, horizon: int = 30) -> pd.DataFrame: ...
        # Returns df[open, high, low, close, volume, amount] for next `horizon` bars
    def predicted_direction(self, bars, horizon=12) -> str: ...   # 'UP' | 'DOWN' | 'FLAT'
    def predicted_range_pct(self, bars, horizon=12) -> float: ... # % expected high-low
    def divergence_score(self, bars, actual_close) -> float: ...  # |predicted - actual| / atr
```

### G.2 — Migration + audit table (2 hours)

`supabase/migrations/2026XXXX_phase_g_kronos.sql`:
```sql
CREATE TABLE IF NOT EXISTS kronos_forecasts (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT NOW(),
  instrument_key TEXT NOT NULL,
  horizon_bars INT NOT NULL,
  timeframe TEXT NOT NULL,
  predicted_close_paisa BIGINT,
  predicted_high_paisa BIGINT,
  predicted_low_paisa BIGINT,
  predicted_direction TEXT CHECK (predicted_direction IN ('UP','DOWN','FLAT')),
  predicted_range_pct NUMERIC,
  raw_forecast JSONB,
  model_version TEXT NOT NULL
);
```

### G.3 — Confluence integration (1 day)

Add `MODEL_FORECAST` dimension to `ConfluenceEngine`:
- Score = +1 if predicted direction matches signal direction, -1 if opposite, 0 if FLAT
- Weight 0.05-0.10 (modest until validated)

### G.4 — Validation harness (3 days)

Run Kronos predictions in shadow mode for 4 weeks alongside live trading:
- Log every prediction + actual outcome to `kronos_forecasts`
- Daily report: directional accuracy, range MAE, calibration histogram
- Compare strategy P&L with-and-without Kronos confluence weight

### G.5 — Decision gate

After 4-week shadow:
- Hit rate > 55% AND useful for ≥ 3 strategies → integrate at weight 0.10+
- Hit rate 50-55% → keep at low weight as tiebreaker
- Hit rate < 50% → drop entirely, save the disk space

---

## Cost estimate

| Item | Cost |
|---|---|
| Model weights download | ~100MB-500MB one-time |
| Inference per call (CPU, 32GB Mac) | ~200-500ms (single instrument, 30-bar horizon) |
| Run for full Top-10 + indices every 5 min | ~5-10s total per cycle |
| RAM footprint | ~1-2GB resident |
| Build effort | ~5-7 days for full G.1-G.5 |
| Ongoing $$ | ₹0 (local inference) |

---

## My recommendation

**Proceed with caution. Build Phase G as SHADOW only — do NOT route any orders through it for first 4 weeks.**

Order of priority vs other open items:
1. (Higher) Finish current paper-forward test of Bloomberg build (4 weeks). Get baseline metrics first.
2. (Higher) Resolve health monitor false positives (still pending).
3. (Equal) Build Kronos shadow mode in parallel with paper test — zero risk, learns in background.
4. (Lower) Backtester rebuild for tick+L2 microstructure replay.

**If we're going to add Kronos, do it now in shadow** so by the end of paper-forward we have real data on whether to keep it.

---

## Quick proof-of-concept command (run when ready)

```bash
pip install torch transformers huggingface_hub pandas numpy
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('NeoQuasar/Kronos-small')
snapshot_download('NeoQuasar/Kronos-Tokenizer-base')
"
# Then clone Kronos repo, copy model/ + examples/prediction_example.py, run
```

---

## Decision needed from user

Pick one:
- **A)** Build Phase G shadow now (parallel to paper test) — 5-7 days
- **B)** Defer until paper-forward complete — revisit in 4 weeks
- **C)** Skip — too uncertain India coverage, focus on existing modules
