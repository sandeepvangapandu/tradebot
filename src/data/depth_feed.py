"""Parse Upstox V3 WebSocket ``mode=full`` depth payloads and compute
L2 microstructure metrics (spread, imbalance, walls).

Upstox V3 full-mode depth path (protobuf decoded to dict by the SDK):
    message["feeds"][<instrument_key>]["fullFeed"]["marketFF"]["marketLevel"]["bidAskQuote"]

Each ``bidAskQuote`` is a list of up to 5 dicts:
    {"bid_q": <bid_qty>, "ask_q": <ask_qty>, "bid_n": <bid_orders>,
     "ask_n": <ask_orders>, "bp": <bid_price_paisa>, "ap": <ask_price_paisa>}

Usage::

    handler = DepthFeedHandler()

    # Called per WebSocket tick
    handler.on_tick(raw_payload)

    # Called once per minute (e.g. from a scheduler)
    rows_written = handler.flush_minute_metrics(minute_ts)

    # Query API for strategies
    metrics = handler.get_latest_metrics("NSE_EQ|INE002A01018", lookback_minutes=5)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------
LevelDict = dict[str, int]   # {price_paisa, qty, orders}
ParsedDepth = dict[str, Any]  # {instrument_key, ts, bids, asks}


# ---------------------------------------------------------------------------
# DepthFeedHandler
# ---------------------------------------------------------------------------

class DepthFeedHandler:
    """Parse, compute metrics, and persist Upstox V3 L2 order-book depth.

    Args:
        db_engine: Sync SQLAlchemy engine for minute-metric writes.
            Falls back to ``src.storage.db.get_sync_engine()`` when ``None``.
        persist_raw: If ``True`` (default), each ``on_tick`` call also writes
            a row to the ``depth_snapshots`` table via the async bulk helper.

    Notes:
        All monetary values are kept in **paisa** (int) internally.
        IST timezone is used for ``minute_ts`` bucketing.
    """

    # IST = UTC+5:30
    _IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60

    def __init__(
        self,
        db_engine=None,
        persist_raw: bool = True,
    ) -> None:
        self._db_engine = db_engine
        self._persist_raw = persist_raw

        # Accumulator: instrument_key → list of per-tick metric snapshots
        # Each snapshot: {spread_bps, bid_qty5, ask_qty5, bid_wall, ask_wall, imbalance}
        self._buffer: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public parse/compute API
    # ------------------------------------------------------------------

    def parse_depth(self, raw_payload: dict) -> ParsedDepth | None:
        """Extract the first instrument's 5-level depth from an Upstox V3 payload.

        The SDK delivers one dict per WebSocket frame. A frame can carry
        multiple instrument feeds; this method iterates all of them and
        returns the **first** instrument whose full-feed depth is present.
        Callers that need all instruments should call ``parse_all_depths``.

        Args:
            raw_payload: Raw decoded WebSocket message dict.

        Returns:
            Dict with keys ``instrument_key``, ``ts`` (UTC-aware datetime),
            ``bids`` and ``asks`` (list of up to 5 level dicts each), or
            ``None`` if the payload does not contain recognisable depth data.
        """
        result = self._parse_first(raw_payload)
        return result

    def parse_all_depths(self, raw_payload: dict) -> list[ParsedDepth]:
        """Extract 5-level depth for *all* instruments in the payload.

        Args:
            raw_payload: Raw decoded WebSocket message dict.

        Returns:
            List of parsed depth dicts (one per instrument with valid depth).
        """
        feeds: dict[str, Any] = raw_payload.get("feeds", {})
        results: list[ParsedDepth] = []
        for instrument_key, feed_data in feeds.items():
            parsed = self._extract_depth(instrument_key, feed_data, raw_payload)
            if parsed is not None:
                results.append(parsed)
        return results

    def compute_spread_bps(self, parsed: ParsedDepth) -> float:
        """Compute best-bid/best-ask spread in basis points.

        Spread (bps) = (best_ask - best_bid) / best_bid * 10_000

        When bid == ask (locked market) the spread is 0.0.
        Returns ``float('inf')`` when either side has no levels.

        Args:
            parsed: Output of ``parse_depth`` / ``parse_all_depths``.

        Returns:
            Spread in basis points as a float.
        """
        bids = parsed.get("bids", [])
        asks = parsed.get("asks", [])

        if not bids or not asks:
            return float("inf")

        best_bid = bids[0]["price_paisa"]
        best_ask = asks[0]["price_paisa"]

        if best_bid <= 0:
            return float("inf")

        if best_bid == best_ask:
            return 0.0

        return (best_ask - best_bid) / best_bid * 10_000

    def compute_imbalance(self, parsed: ParsedDepth, levels: int = 5) -> float:
        """Compute top-N level order-book imbalance.

        Imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

        Range: [-1, +1].  0 = balanced.  +1 = all bid.  -1 = all ask.
        Returns 0.0 when both sides are empty.

        Args:
            parsed: Output of ``parse_depth`` / ``parse_all_depths``.
            levels: Number of top levels to include (default 5).

        Returns:
            Imbalance ratio.
        """
        bids = parsed.get("bids", [])[:levels]
        asks = parsed.get("asks", [])[:levels]

        bid_qty = sum(b["qty"] for b in bids)
        ask_qty = sum(a["qty"] for a in asks)

        total = bid_qty + ask_qty
        if total == 0:
            return 0.0

        return (bid_qty - ask_qty) / total

    def detect_wall(
        self,
        parsed: ParsedDepth,
        side: str = "bid",
        levels: int = 5,
    ) -> int:
        """Find the maximum single-level quantity in the top-N levels (the "wall").

        Args:
            parsed: Output of ``parse_depth`` / ``parse_all_depths``.
            side: ``"bid"`` or ``"ask"``.
            levels: Number of top levels to scan (default 5).

        Returns:
            Max quantity (in shares/lots) at any single level.  0 if empty.
        """
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

        key = "bids" if side == "bid" else "asks"
        book_side = parsed.get(key, [])[:levels]

        if not book_side:
            return 0

        return max(lv["qty"] for lv in book_side)

    # ------------------------------------------------------------------
    # High-frequency tick handler
    # ------------------------------------------------------------------

    def on_tick(self, raw_payload: dict) -> None:
        """Process one raw WebSocket tick.

        Steps:
        1. Parse all instrument depths from the payload.
        2. Compute spread, imbalance, wall metrics.
        3. Accumulate metrics in the per-minute buffer.
        4. Optionally persist the raw snapshot to ``depth_snapshots`` table.

        Args:
            raw_payload: Raw decoded WebSocket message dict.
        """
        all_depths = self.parse_all_depths(raw_payload)

        for parsed in all_depths:
            instrument_key: str = parsed["instrument_key"]

            spread = self.compute_spread_bps(parsed)
            imbalance = self.compute_imbalance(parsed)
            bid_wall = self.detect_wall(parsed, side="bid")
            ask_wall = self.detect_wall(parsed, side="ask")

            bids = parsed.get("bids", [])
            asks = parsed.get("asks", [])
            bid_qty5 = sum(b["qty"] for b in bids[:5])
            ask_qty5 = sum(a["qty"] for a in asks[:5])

            # Accumulate — skip inf spreads (crossed/empty books)
            if spread != float("inf"):
                self._buffer[instrument_key].append(
                    {
                        "spread_bps": spread,
                        "bid_qty5": bid_qty5,
                        "ask_qty5": ask_qty5,
                        "bid_wall": bid_wall,
                        "ask_wall": ask_wall,
                        "imbalance": imbalance,
                    }
                )

            # Persist raw snapshot asynchronously
            if self._persist_raw and parsed.get("bids") is not None:
                self._persist_raw_snapshot(parsed, spread, bid_wall, ask_wall)

    # ------------------------------------------------------------------
    # Minute aggregation
    # ------------------------------------------------------------------

    def flush_minute_metrics(self, minute_ts: datetime) -> int:
        """Aggregate buffered tick metrics and write to ``depth_metrics_minute``.

        Clears the accumulation buffer after writing.  Should be called
        once per minute by a scheduler (e.g. APScheduler or a simple loop).

        Args:
            minute_ts: The minute bucket timestamp (timezone-aware, IST or UTC).
                Stored as-is in the database; use IST-aligned timestamps.

        Returns:
            Number of rows written.
        """
        if not self._buffer:
            return 0

        rows_to_insert: list[dict[str, Any]] = []

        for instrument_key, snapshots in self._buffer.items():
            if not snapshots:
                continue

            n = len(snapshots)
            avg_spread = sum(s["spread_bps"] for s in snapshots) / n
            min_spread = min(s["spread_bps"] for s in snapshots)
            max_spread = max(s["spread_bps"] for s in snapshots)
            avg_bid_qty = int(sum(s["bid_qty5"] for s in snapshots) / n)
            avg_ask_qty = int(sum(s["ask_qty5"] for s in snapshots) / n)
            bid_wall_max = max(s["bid_wall"] for s in snapshots)
            ask_wall_max = max(s["ask_wall"] for s in snapshots)
            imbalance_avg = sum(s["imbalance"] for s in snapshots) / n

            rows_to_insert.append(
                {
                    "instrument_key": instrument_key,
                    "minute_ts": minute_ts,
                    "avg_spread_bps": round(avg_spread, 4),
                    "min_spread_bps": round(min_spread, 4),
                    "max_spread_bps": round(max_spread, 4),
                    "avg_bid_qty_top5": avg_bid_qty,
                    "avg_ask_qty_top5": avg_ask_qty,
                    "bid_wall_max_paisa": bid_wall_max,
                    "ask_wall_max_paisa": ask_wall_max,
                    "imbalance_avg": round(imbalance_avg, 6),
                }
            )

        # Clear buffer before writing so new ticks aren't lost on error
        self._buffer.clear()

        if not rows_to_insert:
            return 0

        engine = self._get_sync_engine()
        sql = text(
            """
            INSERT INTO depth_metrics_minute (
                instrument_key, minute_ts,
                avg_spread_bps, min_spread_bps, max_spread_bps,
                avg_bid_qty_top5, avg_ask_qty_top5,
                bid_wall_max_paisa, ask_wall_max_paisa,
                imbalance_avg
            ) VALUES (
                :instrument_key, :minute_ts,
                :avg_spread_bps, :min_spread_bps, :max_spread_bps,
                :avg_bid_qty_top5, :avg_ask_qty_top5,
                :bid_wall_max_paisa, :ask_wall_max_paisa,
                :imbalance_avg
            )
            ON CONFLICT (instrument_key, minute_ts) DO UPDATE SET
                avg_spread_bps   = EXCLUDED.avg_spread_bps,
                min_spread_bps   = EXCLUDED.min_spread_bps,
                max_spread_bps   = EXCLUDED.max_spread_bps,
                avg_bid_qty_top5 = EXCLUDED.avg_bid_qty_top5,
                avg_ask_qty_top5 = EXCLUDED.avg_ask_qty_top5,
                bid_wall_max_paisa = EXCLUDED.bid_wall_max_paisa,
                ask_wall_max_paisa = EXCLUDED.ask_wall_max_paisa,
                imbalance_avg    = EXCLUDED.imbalance_avg
            """
        )
        with engine.begin() as conn:
            conn.execute(sql, rows_to_insert)

        logger.info(
            "flush_minute_metrics: wrote %d row(s) for minute_ts=%s",
            len(rows_to_insert),
            minute_ts.isoformat(),
        )
        return len(rows_to_insert)

    # ------------------------------------------------------------------
    # Query API for strategies
    # ------------------------------------------------------------------

    def get_latest_metrics(
        self,
        instrument_key: str,
        lookback_minutes: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetch the most recent N minute-metric rows for a given instrument.

        Args:
            instrument_key: Upstox instrument key (e.g. ``"NSE_EQ|INE002A01018"``).
            lookback_minutes: Number of most-recent rows to return (default 5).

        Returns:
            List of dicts with all ``depth_metrics_minute`` columns,
            ordered newest first.  Empty list when no data is found.
        """
        engine = self._get_sync_engine()
        sql = text(
            """
            SELECT
                instrument_key, minute_ts,
                avg_spread_bps, min_spread_bps, max_spread_bps,
                avg_bid_qty_top5, avg_ask_qty_top5,
                bid_wall_max_paisa, ask_wall_max_paisa,
                imbalance_avg
            FROM depth_metrics_minute
            WHERE instrument_key = :instrument_key
            ORDER BY minute_ts DESC
            LIMIT :n
            """
        )
        with engine.connect() as conn:
            result = conn.execute(
                sql,
                {"instrument_key": instrument_key, "n": lookback_minutes},
            )
            rows = result.mappings().all()
        return [dict(r) for r in rows]

    def get_spread_trend(
        self,
        instrument_key: str,
        lookback_minutes: int = 10,
    ) -> dict[str, Any]:
        """Return spread trend summary (mean, min, max over lookback window).

        Args:
            instrument_key: Upstox instrument key.
            lookback_minutes: Look-back window size in minutes.

        Returns:
            Dict with ``mean_spread_bps``, ``min_spread_bps``,
            ``max_spread_bps``, and ``sample_count``.
        """
        rows = self.get_latest_metrics(instrument_key, lookback_minutes)
        if not rows:
            return {
                "mean_spread_bps": None,
                "min_spread_bps": None,
                "max_spread_bps": None,
                "sample_count": 0,
            }

        spreads = [r["avg_spread_bps"] for r in rows if r["avg_spread_bps"] is not None]
        if not spreads:
            return {
                "mean_spread_bps": None,
                "min_spread_bps": None,
                "max_spread_bps": None,
                "sample_count": 0,
            }

        return {
            "mean_spread_bps": sum(spreads) / len(spreads),
            "min_spread_bps": min(spreads),
            "max_spread_bps": max(spreads),
            "sample_count": len(spreads),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_first(self, raw_payload: dict) -> ParsedDepth | None:
        """Return parsed depth for the first instrument in the payload."""
        feeds: dict[str, Any] = raw_payload.get("feeds", {})
        for instrument_key, feed_data in feeds.items():
            parsed = self._extract_depth(instrument_key, feed_data, raw_payload)
            if parsed is not None:
                return parsed
        return None

    def _extract_depth(
        self,
        instrument_key: str,
        feed_data: dict,
        raw_payload: dict,
    ) -> ParsedDepth | None:
        """Extract and normalise depth levels from a single feed entry.

        Args:
            instrument_key: The instrument key string.
            feed_data: The feed sub-dict under ``feeds.<instrument_key>``.
            raw_payload: Full raw payload (used to extract timestamp).

        Returns:
            Parsed depth dict or ``None`` if depth is absent.
        """
        try:
            quotes: list[dict] = (
                feed_data["fullFeed"]["marketFF"]["marketLevel"]["bidAskQuote"]
            )
        except (KeyError, TypeError):
            return None

        if not quotes:
            return None

        bids: list[LevelDict] = []
        asks: list[LevelDict] = []

        for q in quotes:
            bp = q.get("bp", 0)   # bid price (paisa)
            ap = q.get("ap", 0)   # ask price (paisa)
            bq = q.get("bid_q", q.get("bq", 0))  # bid qty
            aq = q.get("ask_q", q.get("aq", 0))  # ask qty
            bn = q.get("bid_n", q.get("bno_b", 0))  # bid orders
            an = q.get("ask_n", q.get("bno_a", 0))  # ask orders

            if bp and bp > 0:
                bids.append({"price_paisa": int(bp), "qty": int(bq), "orders": int(bn)})
            if ap and ap > 0:
                asks.append({"price_paisa": int(ap), "qty": int(aq), "orders": int(an)})

        if not bids and not asks:
            return None

        # Sort bids descending, asks ascending (standard book ordering)
        bids.sort(key=lambda x: x["price_paisa"], reverse=True)
        asks.sort(key=lambda x: x["price_paisa"])

        # Timestamp — prefer feed-level ts, fall back to payload-level
        ts = self._extract_ts(feed_data, raw_payload)

        return {
            "instrument_key": instrument_key,
            "ts": ts,
            "bids": bids[:5],
            "asks": asks[:5],
        }

    @staticmethod
    def _extract_ts(feed_data: dict, raw_payload: dict) -> datetime:
        """Extract a UTC-aware datetime from the payload.

        Tries ``feed_data["fullFeed"]["marketFF"]["ltpc"]["ltt"]`` first,
        then ``raw_payload["currentTs"]`` (ms epoch), then ``now(UTC)``.

        Args:
            feed_data: Feed sub-dict.
            raw_payload: Full raw payload.

        Returns:
            UTC-aware datetime.
        """
        # Try ltt (last trade time) inside fullFeed
        try:
            ltt_ms = feed_data["fullFeed"]["marketFF"]["ltpc"]["ltt"]
            if ltt_ms:
                return datetime.fromtimestamp(int(ltt_ms) / 1000, tz=timezone.utc)
        except (KeyError, TypeError, ValueError):
            pass

        # Try top-level currentTs
        try:
            cts_ms = raw_payload.get("currentTs")
            if cts_ms:
                return datetime.fromtimestamp(int(cts_ms) / 1000, tz=timezone.utc)
        except (TypeError, ValueError):
            pass

        return datetime.now(tz=timezone.utc)

    def _persist_raw_snapshot(
        self,
        parsed: ParsedDepth,
        spread_bps: float,
        bid_wall: int,
        ask_wall: int,
    ) -> None:
        """Fire-and-forget: persist raw depth snapshot to ``depth_snapshots``.

        Uses asyncio to schedule the async bulk insert.  If no running
        event loop is available (e.g. tests), falls back to a new loop.

        Args:
            parsed: Parsed depth dict.
            spread_bps: Pre-computed spread in bps.
            bid_wall: Bid-side wall qty.
            ask_wall: Ask-side wall qty.
        """
        row = {
            "instrument_key": parsed["instrument_key"],
            "ts": parsed["ts"],
            "bids": json.dumps(parsed.get("bids", [])),
            "asks": json.dumps(parsed.get("asks", [])),
            "spread_bps": None if spread_bps == float("inf") else round(spread_bps, 4),
            "bid_wall_top5": bid_wall or None,
            "ask_wall_top5": ask_wall or None,
        }
        try:
            from src.storage.db import bulk_insert_depth  # lazy import

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(bulk_insert_depth([row]))
                else:
                    loop.run_until_complete(bulk_insert_depth([row]))
            except RuntimeError:
                # No event loop in thread — create a temporary one
                asyncio.run(bulk_insert_depth([row]))
        except Exception as exc:
            logger.warning("depth_snapshots insert failed: %s", exc)

    def _get_sync_engine(self):
        """Return the sync engine, preferring the injected one."""
        if self._db_engine is not None:
            return self._db_engine
        from src.storage.db import get_sync_engine  # lazy import
        return get_sync_engine()
