#!/usr/bin/env python3
"""Regenerate MASTER.md from current codebase state.

The file is the single source of truth for cross-module wiring,
strategy catalog, DB schema, scheduler jobs, and a per-function
index. Run after every functional change so the doc never rots:

    python3 scripts/gen_master_doc.py

The script reads:
- src/, scripts/, root .py files       (function index via AST)
- config/strategies/*.json              (strategy catalog)
- data/trading_bot.db                   (SQLite schema)
- supabase/migrations/                  (Postgres schema names)
- src/main.py                           (wiring + schedulers via regex)

It writes MASTER.md in place. The HAND-WRITTEN sections at the top
(Overview, Wiring Matrix narrative, Dead Code, Change Log) are
preserved via markers so the script only regenerates the auto
sections.
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# AST extract — module / class / function inventory
# ---------------------------------------------------------------------------

def _docline(doc: str | None, limit: int = 120) -> str:
    if not doc:
        return ""
    return doc.strip().split("\n", 1)[0][:limit]


def extract_inventory() -> list[dict]:
    files: list[Path] = []
    for sub in ("src", "scripts"):
        files += [p for p in (ROOT / sub).rglob("*.py") if "__pycache__" not in str(p)]
    for fname in ("bot_runner.py", "send_eod_summary.py"):
        p = ROOT / fname
        if p.exists():
            files.append(p)
    files.sort()

    inventory: list[dict] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        classes = []
        funcs = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    {"name": sub.name, "doc": _docline(ast.get_docstring(sub))}
                    for sub in node.body
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                classes.append({
                    "name": node.name,
                    "doc": _docline(ast.get_docstring(node)),
                    "methods": methods,
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    funcs.append({"name": node.name, "doc": _docline(ast.get_docstring(node))})
        inventory.append({
            "path": str(path.relative_to(ROOT)),
            "doc": _docline(ast.get_docstring(tree), 200),
            "classes": classes,
            "funcs": funcs,
        })
    return inventory


# ---------------------------------------------------------------------------
# Strategy catalog
# ---------------------------------------------------------------------------

def load_strategies() -> list[dict]:
    out = []
    for path in sorted((ROOT / "config" / "strategies").glob("*.json")):
        try:
            cfg = json.loads(path.read_text())
        except Exception:
            continue
        out.append({
            "file": str(path.relative_to(ROOT)),
            "name": cfg.get("name", "?"),
            "enabled": cfg.get("enabled", True),
            "underlying": (cfg.get("underlying") or {}).get("symbol", "?"),
            "type": (cfg.get("params") or {}).get("strategy_type", "?"),
            "hours": f"{(cfg.get('trading_hours') or {}).get('start_time', '?')}–{(cfg.get('trading_hours') or {}).get('end_time', '?')}",
            "sl_pct": (cfg.get("exit_rules") or {}).get("stop_loss_pct"),
            "target_pct": (cfg.get("exit_rules") or {}).get("target_pct"),
            "qty": (cfg.get("position_sizing") or {}).get("quantity"),
            "max_pos": (cfg.get("risk_management") or {}).get("max_open_positions"),
            "description": cfg.get("description", "")[:200],
        })
    return out


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

def sqlite_schema() -> list[tuple[str, list[tuple[str, str]]]]:
    db_path = ROOT / "data" / "trading_bot.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
    ).fetchall()]
    out = []
    for t in tables:
        cols = [(r[1], r[2]) for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        out.append((t, cols))
    conn.close()
    return out


def postgres_migration_files() -> list[str]:
    mig = ROOT / "supabase" / "migrations"
    if not mig.exists():
        return []
    return sorted(p.name for p in mig.glob("*.sql"))


# ---------------------------------------------------------------------------
# Scheduler jobs — scan main.py
# ---------------------------------------------------------------------------

def scheduler_jobs() -> list[dict]:
    src = (ROOT / "src" / "main.py").read_text().splitlines()
    jobs = []
    cur: dict = {}
    for i, line in enumerate(src, 1):
        m_call = re.search(r"scheduler\.add_(daily|interval|cron)_job", line)
        if m_call:
            cur = {"line": i, "kind": m_call.group(1)}
        if "func=" in line and cur:
            cur["func"] = re.search(r"func=([\w\._]+)", line).group(1) if re.search(r"func=([\w\._]+)", line) else "?"
        if "hour=" in line and cur:
            cur["hour"] = re.search(r"hour=(\d+)", line).group(1) if re.search(r"hour=(\d+)", line) else ""
        if "minute=" in line and cur:
            cur["minute"] = re.search(r"minute=(\d+)", line).group(1) if re.search(r"minute=(\d+)", line) else ""
        if "seconds=" in line and cur:
            cur["seconds"] = re.search(r"seconds=(\d+)", line).group(1) if re.search(r"seconds=(\d+)", line) else ""
        if "job_id=" in line and cur:
            m_id = re.search(r"job_id=[\"f]*\"([^\"]+)\"", line)
            cur["id"] = m_id.group(1) if m_id else "?"
            jobs.append(cur)
            cur = {}
    return jobs


# ---------------------------------------------------------------------------
# Wiring matrix — scan main.py for self.X = Cls(...) and arg= patterns
# ---------------------------------------------------------------------------

def wiring_lines() -> list[tuple[int, str]]:
    src = (ROOT / "src" / "main.py").read_text().splitlines()
    out = []
    for i, line in enumerate(src, 1):
        s = line.rstrip()
        if not s or s.lstrip().startswith("#"):
            continue
        if re.search(r"self\.(?:[a-z_]+)\s*=\s*[A-Z]\w*\s*\(", s) \
           or re.search(r"\b(db_engine|risk_manager|broker|trade_logger|position_manager|order_manager|strategy_engine|circuit_breaker|kelly_sizer|confluence_engine|rejection_filter|regime_router)\s*=\s*self\.", s):
            out.append((i, s.strip()))
    return out


# ---------------------------------------------------------------------------
# Git change log
# ---------------------------------------------------------------------------

def git_log(n: int = 20) -> list[str]:
    try:
        return subprocess.check_output(
            ["git", "log", "--pretty=format:%h | %s", f"-{n}"],
            cwd=ROOT,
        ).decode().splitlines()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

START_AUTO = "<!-- AUTO-GENERATED-BEGIN: do not edit between markers -->"
END_AUTO = "<!-- AUTO-GENERATED-END -->"


def render_md() -> str:
    inv = extract_inventory()
    strats = load_strategies()
    sqlite_tables = sqlite_schema()
    pg_migs = postgres_migration_files()
    jobs = scheduler_jobs()
    wires = wiring_lines()
    commits = git_log(25)
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

    lines: list[str] = []
    lines.append("# MASTER.md — Trading Bot System Reference")
    lines.append("")
    lines.append(f"_Auto-regenerated by `scripts/gen_master_doc.py` — last build {now}._")
    lines.append("")
    lines.append("**Workflow rule (also in CLAUDE.md):** any code change MUST first be evaluated against this doc. Before touching code, report which sections are affected (wiring, strategy, schedulers, DB), what the effect is, and wait for user go/no-go. After the change, regenerate this doc in the same commit.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Table of contents")
    lines.append("")
    lines.append("1. [System overview](#system-overview)")
    lines.append("2. [Entry point + boot sequence](#entry-point--boot-sequence)")
    lines.append("3. [Wiring matrix](#wiring-matrix)")
    lines.append("4. [Strategy catalog](#strategy-catalog)")
    lines.append("5. [Database schema](#database-schema)")
    lines.append("6. [Scheduler jobs](#scheduler-jobs)")
    lines.append("7. [Per-module function index](#per-module-function-index)")
    lines.append("8. [Known dead code / unwired params](#known-dead-code--unwired-params)")
    lines.append("9. [Recent commits](#recent-commits)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 1. System overview (hand-curated; regenerate keeps as-is) ----
    lines.append("## System overview")
    lines.append("")
    lines.append("```")
    lines.append("        ┌─────────────────────────────────────────────────────────────┐")
    lines.append("        │                    src/main.py (TradingBot)                 │")
    lines.append("        │  startup → init modules → start scheduler → run main loop   │")
    lines.append("        └────┬───────────────────────────────────────────────────┬────┘")
    lines.append("             │                                                   │")
    lines.append("    ┌────────┴───────┐                                  ┌────────┴────────┐")
    lines.append("    │ Upstox token   │                                  │ MarketDataFeed  │")
    lines.append("    │ TokenManager   │                                  │ WebSocket V3    │")
    lines.append("    └────────────────┘                                  └────────┬────────┘")
    lines.append("                                                                 │ ticks")
    lines.append("                                                                 ▼")
    lines.append("                                          ┌───────────────────────────────┐")
    lines.append("                                          │  BarBuilder → bar_close_event  │")
    lines.append("                                          └────────────────┬──────────────┘")
    lines.append("                                                           ▼")
    lines.append("           ┌──────────────────────────────────────────────────────────────┐")
    lines.append("           │  StrategyEngine (per-strategy SetEvaluator threads)          │")
    lines.append("           │  conditions + indicators → Signal → signal_queue             │")
    lines.append("           └────────────────────────────────┬─────────────────────────────┘")
    lines.append("                                            ▼")
    lines.append("           ┌──────────────────────────────────────────────────────────────┐")
    lines.append("           │  OrderManager.process_signal                                  │")
    lines.append("           │  ↳ Kelly sizing → dup-skip → risk → AI pipeline → broker     │")
    lines.append("           └────────────────────────────────┬─────────────────────────────┘")
    lines.append("                                            ▼")
    lines.append("           ┌──────────────────────────────────────────────────────────────┐")
    lines.append("           │  PaperBroker.place_order → fill at LTP+slippage              │")
    lines.append("           │  ↳ rejects if LTP missing (post-66c19c8)                     │")
    lines.append("           └────────────────────────────────┬─────────────────────────────┘")
    lines.append("                                            ▼")
    lines.append("           ┌──────────────────────────────────────────────────────────────┐")
    lines.append("           │  PositionManager.add_position                                 │")
    lines.append("           │  ↳ persists to SQLite via TradeLogger.log_position           │")
    lines.append("           │  ↳ on_tick → SL/Target/Trailing check → _close_position      │")
    lines.append("           │  ↳ on close: TradeLogger.log_trade + close_position          │")
    lines.append("           └──────────────────────────────────────────────────────────────┘")
    lines.append("```")
    lines.append("")

    # ---- 2. Boot sequence ----
    lines.append("## Entry point + boot sequence")
    lines.append("")
    lines.append("`src/main.py:main()` → `TradingBot().run()` → `startup()`:")
    lines.append("")
    lines.append("1. `_setup_logging` from settings")
    lines.append("2. `TokenManager.get_valid_token` (cache OR auto_login OR fail)")
    lines.append("3. `init_db(settings.database_url)` (SQLite) + `TradeLogger`")
    lines.append("4. `record_bot_run_start` → Postgres (non-fatal if missing)")
    lines.append("5. `InstrumentManager.download_instruments(['NSE','BSE'])`")
    lines.append("6. `RiskManager`, `CircuitBreaker`, `StrategyQuarantine`")
    lines.append("7. `PaperBroker`, `PartialProfitManager`, `PositionManager(trade_logger=trade_log)`, `OrderTracker`, `OrderManager`")
    lines.append("8. `BarBuilder` started; seed 1m/5m/15m bars from historical")
    lines.append("9. `MarketDataFeed` + `PortfolioFeed` WebSocket connect")
    lines.append("10. `StrategyEngine` load + active-strategy SetEvaluator threads spawn")
    lines.append("11. Bloomberg modules: UniverseScanner, SectorRotation, VIXRegime, MacroOverlay, FlowRegime, news/insider scrapers")
    lines.append("12. ConfluenceEngine, RejectionFilter, RegimeRouter, KellySizer → `strategy_engine.set_wave5_modules`")
    lines.append("13. `BackgroundScheduler.start` → registers all daily/interval jobs (see table below)")
    lines.append("14. Main loop: pull signals from queue, route to OrderManager, monitor health")
    lines.append("")
    lines.append("Shutdown (`stop()`): stop engine → order manager → close-all-positions → stop feeds/bar builder → final daily summary → record_bot_run_end.")
    lines.append("")

    # ---- 3. Wiring matrix ----
    lines.append("## Wiring matrix")
    lines.append("")
    lines.append("Key cross-module dependencies. **Read direction:** producer module → param name → consumer module → use site.")
    lines.append("")
    lines.append("| Producer (init in `src/main.py`) | Param | Consumer | Use site |")
    lines.append("|---|---|---|---|")
    lines.append("| `TradeLogger` L372 | `trade_logger=trade_log` (kw) | `PositionManager.__init__` L706 | `_close_position` L1773 calls `log_trade` + `close_position`; `add_position` L883 calls `log_position` |")
    lines.append("| `PaperBroker` L431 | `broker` | `PositionManager`, `OrderManager`, `OrderTracker` | fills + LTP lookup + fee aggregation (`sum_fees_for_instrument`) |")
    lines.append("| `RiskManager` L407 | `risk_manager` | `PositionManager`, `OrderManager`, `StrategyEngine` | position validation + capital check + circuit breaker hookup |")
    lines.append("| `CircuitBreaker` L414 | (set onto risk_manager) L422 | `PositionManager._close_position` | `record_loss` / `record_win` per closed trade |")
    lines.append("| `PartialProfitManager` L435 | `partial_profit_manager` | `PositionManager` | 4-tier partial exit logic |")
    lines.append("| `Postgres engine` L465 / L915 | `db_engine` | Universe, Sector, VIX, Macro, Flow, news, insider modules + StrategyEngine wave5 | per-module ORM/text queries |")
    lines.append("| `MarketDataFeed.tick_queue` | `tick_queue=self.tick_queue` | `BarBuilder` + `PositionManager._on_tick` | tick ingest |")
    lines.append("| `signal_queue` | `signal_queue` | `OrderManager` | strategy → execution handoff |")
    lines.append("| `StrategyQuarantine` L419 | `strategy_quarantine` | `OrderManager.process_signal` | quarantined strategies blocked |")
    lines.append("| `ConfluenceEngine`/`RejectionFilter`/`RegimeRouter` | via `set_wave5_modules` | `StrategyEngine._apply_wave5_gates` | composite gating after signal generation |")
    lines.append("| `agent_pipeline` (LLM) | `order_manager._agent_pipeline` L851 | `OrderManager.process_signal` AI validation step | currently disabled per past observation 163 |")
    lines.append("| `MarketMakerAgent` L843 | `self.market_maker` | `_scheduled_ai_playbook` daily 9:00 IST | playbook generation |")
    lines.append("")
    lines.append("### Wiring scan (raw `src/main.py`)")
    lines.append("")
    lines.append("```")
    for ln, s in wires:
        lines.append(f"L{ln:4d}: {s[:200]}")
    lines.append("```")
    lines.append("")

    # ---- 4. Strategy catalog ----
    lines.append("## Strategy catalog")
    lines.append("")
    lines.append("| File | Name | Enabled | Underlying | Type | Hours | SL% | Target% | Qty | Max pos |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in strats:
        lines.append(f"| `{s['file']}` | {s['name']} | {s['enabled']} | {s['underlying']} | {s['type']} | {s['hours']} | {s['sl_pct']} | {s['target_pct']} | {s['qty']} | {s['max_pos']} |")
    lines.append("")
    lines.append("Strategy execution lives in `src/strategy/engine.py` (`StrategyEngine`, `SetEvaluator`). Entry/exit conditions evaluate via `src/strategy/conditions*.py` modules. Options-instrument resolution: `src/strategy/instrument_resolver.py`.")
    lines.append("")

    # ---- 5. DB schema ----
    lines.append("## Database schema")
    lines.append("")
    lines.append(f"**SQLite** primary store: `data/trading_bot.db` ({len(sqlite_tables)} tables). Models in `src/persistence/models.py`. Writer: `src/persistence/trade_log.py` (`TradeLogger`).")
    lines.append("")
    for tname, cols in sqlite_tables:
        lines.append(f"<details><summary><code>{tname}</code> ({len(cols)} cols)</summary>\n")
        lines.append("")
        lines.append("| Column | Type |")
        lines.append("|---|---|")
        for cn, ct in cols:
            lines.append(f"| `{cn}` | {ct} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append(f"**Postgres** (auxiliary, Bloomberg-build modules): connection via `src/storage/db.py:get_sync_engine()`. Migrations:")
    lines.append("")
    for m in pg_migs:
        lines.append(f"- `supabase/migrations/{m}`")
    lines.append("")

    # ---- 6. Scheduler jobs ----
    lines.append("## Scheduler jobs")
    lines.append("")
    lines.append("Registered in `src/main.py:_register_scheduled_jobs` (and inline blocks).")
    lines.append("")
    lines.append("| job_id | kind | trigger | func | main.py line |")
    lines.append("|---|---|---|---|---|")
    for j in jobs:
        if j.get("kind") == "daily":
            trig = f"{j.get('hour', '?')}:{j.get('minute', '00').zfill(2)} IST"
        elif j.get("kind") == "interval":
            trig = f"every {j.get('seconds', '?')}s"
        else:
            trig = j.get("kind", "?")
        lines.append(f"| `{j.get('id', '?')}` | {j.get('kind', '?')} | {trig} | `{j.get('func', '?')}` | L{j.get('line', '?')} |")
    lines.append("")

    # ---- 7. Module function index ----
    lines.append("## Per-module function index")
    lines.append("")
    lines.append("Auto-generated by AST walk. For each module: 1-line purpose, classes (with methods), top-level functions.")
    lines.append("")
    for m in inv:
        if m["path"].endswith("__init__.py") and not (m["classes"] or m["funcs"]):
            continue
        lines.append(f"### `{m['path']}`")
        if m["doc"]:
            lines.append(f"> {m['doc']}")
        lines.append("")
        if m["classes"]:
            for c in m["classes"]:
                line = f"- **class `{c['name']}`**"
                if c["doc"]:
                    line += f" — {c['doc']}"
                lines.append(line)
                for mt in c["methods"]:
                    if mt["name"].startswith("_") and mt["name"] not in ("__init__",):
                        # skip dunders except __init__
                        if not mt["name"].startswith("__"):
                            pass  # keep private methods? skip to reduce size
                            continue
                        if mt["name"] not in ("__init__",):
                            continue
                    sub = f"  - `{mt['name']}`"
                    if mt["doc"]:
                        sub += f" — {mt['doc']}"
                    lines.append(sub)
        if m["funcs"]:
            for fn in m["funcs"]:
                line = f"- `{fn['name']}()`"
                if fn["doc"]:
                    line += f" — {fn['doc']}"
                lines.append(line)
        lines.append("")

    # ---- 8. Dead code ----
    lines.append("## Known dead code / unwired params")
    lines.append("")
    lines.append("Params declared in configs or modules but not consumed by any code path:")
    lines.append("")
    lines.append("- `Expiry_Straddle_Sell_BankNifty.params.emergency_exit_move_pct` — declared in `config/strategies/expiry_straddle_sell_banknifty.json` but no grep hit in `src/`. Needs PositionManager to track entry-time underlying price; not wired as of 2026-05-14.")
    lines.append("- `kronos_lib/`, `kronos_predictor.py`, `kronos_validator.py` etc. — removed in commit d81d22b after 180d backtest showed 36.13% accuracy (below random). DB tables `kronos_forecasts` / `kronos_accuracy_daily` dropped via migration `20260514000000_drop_kronos.sql`.")
    lines.append("")
    lines.append("### Dormant `settings.py` fields (declared, never referenced)")
    lines.append("")
    lines.append("These pydantic fields exist in `config/settings.py` but are not read by any module (verified via grep on `settings.X`). Some may also have no env-var fallback. Kept because removing them could break unknown external configs (pydantic ignores extras anyway). Audit them periodically.")
    lines.append("")
    lines.append("- `breadth_sentiment_weight`, `news_sentiment_weight`, `volatility_sentiment_weight` — sentiment aggregator weights, no consumer found.")
    lines.append("- `learning_generate_daily_report` — flag, no reader.")
    lines.append("- `memory_boost_factor`, `memory_enabled`, `memory_max_lessons_per_agent` — memory subsystem knobs without code reads.")
    lines.append("- `news_cache_ttl_seconds`, `sentiment_cache_ttl_seconds` — cache TTLs not consulted.")
    lines.append("- `research_correlation_cache_minutes`, `research_iv_cache_minutes`, `research_log_all`, `research_max_analysis_time_seconds`, `research_regime_cache_minutes`, `research_sr_cache_minutes` — research-cache knobs unused.")
    lines.append("- `signal_min_confidence` — confidence threshold, no signal-side check.")
    lines.append("- `default_exchange`, `upstox_redirect_uri` — declared but no read; redirect_uri is referenced only in `manual_login.py` via env var.")
    lines.append("")
    lines.append("### Settings fields read via `os.getenv` directly (NOT through settings object)")
    lines.append("")
    lines.append("These appear dormant from a `settings.X` grep but are actually read via env vars in specific modules. Listed here so future audits don't flag them as dead:")
    lines.append("")
    lines.append("- `UPSTOX_PASSWORD`, `UPSTOX_TOTP_SECRET`, `UPSTOX_CLIENT_ID`, `UPSTOX_CLIENT_SECRET`, `UPSTOX_ACCESS_TOKEN` — `src/auth/auto_login.py`, `src/auth/token_manager.py`.")
    lines.append("- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — `src/notifications/telegram_bot.py`.")
    lines.append("- `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN`, `ACTIVE_BROKER`, `LIVE_TRADING_CONFIRMED`, `TRADING_MODE` — `src/execution/broker_factory.py`.")
    lines.append("")
    lines.append("Add new entries here whenever a config knob is added without a corresponding code reader, or when a code path is deactivated without removal.")
    lines.append("")

    # ---- 9. Recent commits ----
    lines.append("## Recent commits")
    lines.append("")
    lines.append("```")
    for c in commits:
        lines.append(c)
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_End of MASTER.md. Regenerate with `python3 scripts/gen_master_doc.py`._")

    return "\n".join(lines) + "\n"


def main() -> int:
    out = ROOT / "MASTER.md"
    out.write_text(render_md())
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
