"""Q-learning exit agent for options positions.

State: 8 normalized features. Actions: HOLD / EXIT / TIGHTEN_SL.
Trains incrementally from paper trade trajectories stored in trading_bot.db.
Falls back to rule-based policy until enough trajectories exist (< 50 trades).

All monetary values are in PAISA (1/100 INR) as integers to avoid floating-point
issues. Prices are converted to rupees only for display / logging.

Serialisation: Q-table is pickled to ``models/rl_exit_agent.pkl``. The file is
self-contained so the agent can be restored across bot restarts without touching
the database schema.
"""

from __future__ import annotations

import math
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Action space
ACTION_HOLD = "HOLD"
ACTION_EXIT = "EXIT"
ACTION_TIGHTEN_SL = "TIGHTEN_SL"
_ACTIONS: list[str] = [ACTION_HOLD, ACTION_EXIT, ACTION_TIGHTEN_SL]
_ACTION_IDX: dict[str, int] = {a: i for i, a in enumerate(_ACTIONS)}

# State features (must match _build_state_vector order)
_FEATURE_NAMES: list[str] = [
    "pnl_pct",
    "bars_held_norm",
    "momentum",
    "vol_regime",
    "dist_to_sl",
    "dist_to_target",
    "trailing_active",
    "peak_gain_norm",
]

# Discretisation: 5 bins per feature  →  key space = 5^8 = 390 625
_N_BINS: int = 5

# Training threshold: use RL policy only after this many complete episodes
_TRAIN_THRESHOLD: int = 50

# Max bars to hold before bars_held_norm saturates at 1.0  (4-hour session ÷ 5min bars)
_MAX_BARS: int = 48

# TIGHTEN_SL brings stop 30 % closer to current price
_TIGHTEN_RATIO: float = 0.30

# Reward shaping constants
_EXIT_LOSS_PENALTY: float = 0.5   # extra penalty on top of the raw loss
_HOLD_PNL_SCALE: float = 1.0      # weight for incremental PnL reward during hold

# Fallback policy thresholds
_FALLBACK_STOP_PCT: float = -0.5    # exit if PnL < -50 % of premium
_FALLBACK_TARGET_PCT: float = 0.8   # exit if PnL > +80 % of premium
_FALLBACK_TIGHTEN_MOMENTUM: float = -0.3  # tighten SL if trailing active + momentum weak


# ---------------------------------------------------------------------------
# Helper: bin edges for each feature
# ---------------------------------------------------------------------------

def _make_bin_edges() -> list[np.ndarray]:
    """Return bin edges for each of the 8 state features.

    Edges define the boundaries for ``np.digitize``.  The resulting bin index
    is in [1, _N_BINS]; clamp to [0, _N_BINS - 1] to get a 0-based index.
    """
    edges: list[np.ndarray] = []

    # 1. pnl_pct  [-2, 2] → 5 equal bins
    edges.append(np.linspace(-2.0, 2.0, _N_BINS - 1))

    # 2. bars_held_norm [0, 1] → 5 equal bins
    edges.append(np.linspace(0.0, 1.0, _N_BINS - 1))

    # 3. momentum [-1, 1] → 5 equal bins
    edges.append(np.linspace(-1.0, 1.0, _N_BINS - 1))

    # 4. vol_regime [0, 3] → [0, 0.8, 1.0, 1.5, 3.0]
    edges.append(np.array([0.8, 1.0, 1.5, 3.0]))

    # 5. dist_to_sl [0, 1] → 5 equal bins
    edges.append(np.linspace(0.0, 1.0, _N_BINS - 1))

    # 6. dist_to_target [0, 1] → 5 equal bins
    edges.append(np.linspace(0.0, 1.0, _N_BINS - 1))

    # 7. trailing_active {0, 1} → 2-value discrete mapped to 5 bins
    #    0.0 → bin 0; 1.0 → bin 4
    edges.append(np.array([0.25, 0.5, 0.75]))

    # 8. peak_gain_norm [0, 1] → 5 equal bins
    edges.append(np.linspace(0.0, 1.0, _N_BINS - 1))

    return edges


_BIN_EDGES: list[np.ndarray] = _make_bin_edges()


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------

class RLExitAgent:
    """Tabular Q-learning agent for deciding when to exit an options position.

    The agent observes an 8-dimensional state on every tick and returns one of
    three actions: HOLD, EXIT, or TIGHTEN_SL.  Q-values are stored in a plain
    Python dict keyed by a discretised state tuple, making the implementation
    dependency-free (stdlib + numpy only, no ML frameworks).

    Training is online: call ``record_step`` after each tick.  The agent uses
    an epsilon-greedy policy once ``is_trained`` returns True (>= 50 complete
    episodes).  Before that it falls back to a deterministic rule-based policy
    so the bot is never paralysed waiting for data.

    Thread safety: A ``threading.RLock`` guards the Q-table and episode counter
    so ``get_action`` and ``record_step`` can be called from concurrent threads
    (tick handler + position monitor).

    Args:
        db_path: Absolute path to ``trading_bot.db`` (SQLite).  Currently used
            for future trajectory replay; the online update path is independent.
        learning_rate: Alpha for Bellman update (default 0.1).
        discount: Gamma for future reward discounting (default 0.95).
        epsilon: Exploration rate for epsilon-greedy policy (default 0.1).
    """

    def __init__(
        self,
        db_path: str,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        self._db_path = db_path
        self._lr = learning_rate
        self._gamma = discount
        self._epsilon = epsilon

        # Q-table: state_key → [q_hold, q_exit, q_tighten_sl]
        self._q_table: dict[tuple[int, ...], list[float]] = {}

        # Episode and step tracking
        self._episode_count: int = 0   # complete positions closed
        self._step_count: int = 0      # total tick-level transitions seen

        # Protect shared state
        self._lock = threading.RLock()

        # RNG for epsilon-greedy exploration
        self._rng = np.random.default_rng(seed=42)

        logger.info(
            "RLExitAgent initialised | lr={} discount={} epsilon={} db={}",
            learning_rate, discount, epsilon, db_path,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_action(self, state: dict[str, float]) -> str:
        """Choose an action given the current position state.

        Uses epsilon-greedy over the Q-table when ``is_trained`` is True;
        otherwise delegates to the deterministic fallback policy.

        Args:
            state: Dict with keys matching ``_FEATURE_NAMES``.  All values
                must be real numbers in their documented ranges.

        Returns:
            One of ``'HOLD'``, ``'EXIT'``, or ``'TIGHTEN_SL'``.
        """
        with self._lock:
            if not self.is_trained:
                action = self._fallback_policy(state)
                logger.debug(
                    "RLExitAgent fallback | episodes={} action={}",
                    self._episode_count, action,
                )
                return action

            key = self._state_to_key(state)

            # Epsilon-greedy: explore with probability epsilon
            if self._rng.random() < self._epsilon:
                action = _ACTIONS[int(self._rng.integers(len(_ACTIONS)))]
                logger.debug("RLExitAgent explore | action={}", action)
                return action

            q_values = self._get_q_values(key)
            action = _ACTIONS[int(np.argmax(q_values))]
            logger.debug(
                "RLExitAgent greedy | key={} q={} action={}",
                key, [round(v, 4) for v in q_values], action,
            )
            return action

    def record_step(
        self,
        state: dict[str, float],
        action: str,
        reward: float,
        next_state: dict[str, float],
        done: bool,
    ) -> None:
        """Perform an online Q-table update after one tick.

        Applies the standard Bellman update:
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') − Q(s, a)]

        When ``done`` is True the episode counter is incremented; once the
        counter reaches ``_TRAIN_THRESHOLD`` the agent switches from fallback
        to the learned policy.

        Args:
            state: Feature dict for the current tick.
            action: Action taken (``'HOLD'``, ``'EXIT'``, ``'TIGHTEN_SL'``).
            reward: Scalar reward signal (see ``compute_reward``).
            next_state: Feature dict for the next tick.
            done: True if the position was closed at this step.
        """
        with self._lock:
            key = self._state_to_key(state)
            next_key = self._state_to_key(next_state)

            q_values = self._get_q_values(key)
            next_q_values = self._get_q_values(next_key)

            a_idx = _ACTION_IDX.get(action, 0)

            # Bellman update
            current_q = q_values[a_idx]
            target = reward + (0.0 if done else self._gamma * float(np.max(next_q_values)))
            updated_q = current_q + self._lr * (target - current_q)

            q_values[a_idx] = updated_q
            self._q_table[key] = q_values

            self._step_count += 1

            if done:
                self._episode_count += 1
                logger.info(
                    "RLExitAgent episode complete | total_episodes={} steps_total={}",
                    self._episode_count, self._step_count,
                )

    def compute_reward(
        self,
        pnl_change_pct: float,
        action: str,
        done: bool,
        final_pnl_pct: float,
    ) -> float:
        """Calculate the scalar reward for a tick-level transition.

        Reward design:
        - **HOLD**: incremental PnL change scaled by ``_HOLD_PNL_SCALE``.
          Encourages holding when trend is positive, discourages holding
          through drawdown.
        - **EXIT** (profitable): final PnL percentage.  Exiting at a strong
          profit is rewarded proportionally.
        - **EXIT** (loss): final PnL minus an additional penalty to discourage
          panic exits that realise unnecessary losses.
        - **TIGHTEN_SL**: same as HOLD for the current tick; the SL tightening
          itself does not produce an immediate PnL event.

        Args:
            pnl_change_pct: Change in position PnL this tick as a fraction of
                max premium (e.g. 0.02 = +2 % this bar).
            action: Action taken at this step.
            done: True if the position is now closed.
            final_pnl_pct: Final PnL as fraction of max premium on position
                close (relevant only when ``done`` is True).

        Returns:
            Scalar reward value.
        """
        if action == ACTION_EXIT:
            if done:
                if final_pnl_pct >= 0.0:
                    return float(final_pnl_pct)
                else:
                    # Loss: penalise beyond the raw loss to discourage early cuts
                    return float(final_pnl_pct) - _EXIT_LOSS_PENALTY
            # EXIT called but done=False (should not happen; guard anyway)
            return 0.0

        if action == ACTION_HOLD or action == ACTION_TIGHTEN_SL:
            return float(pnl_change_pct) * _HOLD_PNL_SCALE

        # Unknown action: neutral
        logger.warning("RLExitAgent.compute_reward: unknown action '{}'", action)
        return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise Q-table and counters to a pickle file.

        Args:
            path: Destination file path (e.g. ``'models/rl_exit_agent.pkl'``).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "q_table": self._q_table,
            "episode_count": self._episode_count,
            "step_count": self._step_count,
            "lr": self._lr,
            "gamma": self._gamma,
            "epsilon": self._epsilon,
        }
        with dest.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "RLExitAgent saved | path={} q_states={} episodes={}",
            path, len(self._q_table), self._episode_count,
        )

    def load(self, path: str) -> None:
        """Restore Q-table and counters from a pickle file.

        Args:
            path: Source file path written by ``save``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is missing required keys.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"RLExitAgent checkpoint not found: {path}")

        with src.open("rb") as fh:
            payload: dict[str, Any] = pickle.load(fh)

        required = {"q_table", "episode_count", "step_count"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"Checkpoint missing keys: {missing}")

        with self._lock:
            self._q_table = payload["q_table"]
            self._episode_count = payload["episode_count"]
            self._step_count = payload["step_count"]
            # Optionally restore hyper-params if present
            if "lr" in payload:
                self._lr = payload["lr"]
            if "gamma" in payload:
                self._gamma = payload["gamma"]
            if "epsilon" in payload:
                self._epsilon = payload["epsilon"]

        logger.info(
            "RLExitAgent loaded | path={} q_states={} episodes={}",
            path, len(self._q_table), self._episode_count,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True once the agent has completed at least 50 trading episodes.

        Below this threshold ``get_action`` defers to the rule-based fallback
        policy so the bot is never blocked waiting for learning data.
        """
        return self._episode_count >= _TRAIN_THRESHOLD

    @property
    def episode_count(self) -> int:
        """Number of complete position episodes seen so far."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Total tick-level transitions processed."""
        return self._step_count

    @property
    def q_table_size(self) -> int:
        """Number of distinct state entries in the Q-table."""
        return len(self._q_table)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_to_key(self, state: dict[str, float]) -> tuple[int, ...]:
        """Discretise 8 continuous features into a Q-table lookup key.

        Each feature is mapped to a bin index in [0, ``_N_BINS`` − 1] using
        the pre-computed edges in ``_BIN_EDGES``.  Missing or NaN features
        are treated as 0 (lowest bin) with a warning so the bot never crashes
        on incomplete tick data.

        Args:
            state: Feature dict (keys defined in ``_FEATURE_NAMES``).

        Returns:
            Tuple of ``len(_FEATURE_NAMES)`` ints, each in [0, ``_N_BINS``−1].
        """
        indices: list[int] = []
        for feature, edges in zip(_FEATURE_NAMES, _BIN_EDGES):
            raw = state.get(feature)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                logger.warning(
                    "RLExitAgent: missing/NaN feature '{}' — defaulting to bin 0",
                    feature,
                )
                indices.append(0)
                continue

            value = float(raw)
            # np.digitize returns 1-based index; clamp to [0, _N_BINS - 1]
            bin_idx = int(np.digitize(value, edges))
            bin_idx = max(0, min(_N_BINS - 1, bin_idx))
            indices.append(bin_idx)

        return tuple(indices)

    def _get_q_values(self, key: tuple[int, ...]) -> list[float]:
        """Retrieve Q-values for a state key, initialising to zeros if unseen.

        Args:
            key: Discretised state tuple from ``_state_to_key``.

        Returns:
            Mutable list ``[q_hold, q_exit, q_tighten_sl]``.
        """
        if key not in self._q_table:
            self._q_table[key] = [0.0, 0.0, 0.0]
        return self._q_table[key]

    def _fallback_policy(self, state: dict[str, float]) -> str:
        """Deterministic rule-based policy used before sufficient training data.

        Rules (evaluated in order; first match wins):
        1. ``pnl_pct < -0.5`` → EXIT  (hard stop, protect capital).
        2. ``pnl_pct > +0.8`` → EXIT  (lock in profit near full premium).
        3. ``trailing_active == 1.0`` and ``momentum < -0.3`` → TIGHTEN_SL
           (momentum fading while trailing is live; ratchet the stop).
        4. Default → HOLD.

        Args:
            state: Feature dict.

        Returns:
            Action string.
        """
        pnl_pct: float = float(state.get("pnl_pct", 0.0))
        momentum: float = float(state.get("momentum", 0.0))
        trailing_active: float = float(state.get("trailing_active", 0.0))

        if pnl_pct < _FALLBACK_STOP_PCT:
            return ACTION_EXIT

        if pnl_pct > _FALLBACK_TARGET_PCT:
            return ACTION_EXIT

        if trailing_active >= 1.0 and momentum < _FALLBACK_TIGHTEN_MOMENTUM:
            return ACTION_TIGHTEN_SL

        return ACTION_HOLD

    # ------------------------------------------------------------------
    # Trajectory replay (future use / offline warm-start)
    # ------------------------------------------------------------------

    def replay_from_db(self, n_episodes: int = 200) -> int:
        """Warm-start the Q-table from closed trades stored in ``trading_bot.db``.

        Reads the ``trades`` table, reconstructs approximate state transitions
        from entry/exit prices and ATR, and applies batch Q-updates.  This is
        called once at bot startup when a valid checkpoint does not exist so the
        agent can bootstrap from historical paper trades without waiting for new
        live sessions.

        Note: The state reconstruction is a heuristic approximation because full
        per-tick trajectory data is not stored in SQLite; only the final P&L and
        a small set of trade metadata are available.  The result is a rough prior
        that the online updates then refine.

        Args:
            n_episodes: Maximum number of closed trades to replay (most recent
                first).

        Returns:
            Number of trade rows successfully replayed.
        """
        if not Path(self._db_path).exists():
            logger.warning(
                "RLExitAgent.replay_from_db: DB not found at '{}', skipping warm-start",
                self._db_path,
            )
            return 0

        try:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT strategy, instrument_key, entry_price, exit_price, quantity,
                       pnl_paisa, duration_seconds, stop_loss_price, target_price,
                       closed_at
                FROM trades
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                (n_episodes,),
            )
            rows = cur.fetchall()
            conn.close()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "RLExitAgent.replay_from_db: query failed ({}), skipping warm-start",
                exc,
            )
            return 0

        replayed = 0
        for row in rows:
            try:
                self._replay_single_trade(row)
                replayed += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "RLExitAgent.replay_from_db: skipping row — {}",
                    exc,
                )

        logger.info(
            "RLExitAgent warm-start complete | replayed={} q_states={}",
            replayed, len(self._q_table),
        )
        return replayed

    def _replay_single_trade(self, row: sqlite3.Row) -> None:
        """Synthesise and apply a Q-update from a single closed trade row.

        Constructs a mock terminal state from the trade outcome and treats the
        entire trade as a single (s, a, r, s') transition with ``done=True``.

        Args:
            row: SQLite row from the ``trades`` table.
        """
        entry_price: int = int(row["entry_price"] or 0)
        exit_price: int = int(row["exit_price"] or 0)
        pnl_paisa: int = int(row["pnl_paisa"] or 0)
        duration_s: int = int(row["duration_seconds"] or 0)
        sl_price: int = int(row["stop_loss_price"] or 0)
        target_price: int = int(row["target_price"] or 0)

        if entry_price <= 0:
            return

        # Approximate premium as entry price (options: cost of the premium)
        max_premium: int = entry_price

        # Final PnL as fraction of premium
        final_pnl_pct = pnl_paisa / max_premium if max_premium else 0.0
        final_pnl_pct = float(np.clip(final_pnl_pct, -2.0, 2.0))

        # Bars held: assume 5-min bars
        bars_held = duration_s / 300.0
        bars_held_norm = float(np.clip(bars_held / _MAX_BARS, 0.0, 1.0))

        # Distance to SL and target at exit (terminal state → both 0)
        dist_to_sl = 0.0
        dist_to_target = 0.0

        # Heuristic trailing active: assume yes if profit > 30 %
        trailing_active = 1.0 if final_pnl_pct > 0.3 else 0.0

        terminal_state: dict[str, float] = {
            "pnl_pct": final_pnl_pct,
            "bars_held_norm": bars_held_norm,
            "momentum": 0.0,      # unknown at replay time
            "vol_regime": 1.0,    # neutral assumption
            "dist_to_sl": dist_to_sl,
            "dist_to_target": dist_to_target,
            "trailing_active": trailing_active,
            "peak_gain_norm": max(0.0, final_pnl_pct),
        }

        reward = self.compute_reward(
            pnl_change_pct=0.0,
            action=ACTION_EXIT,
            done=True,
            final_pnl_pct=final_pnl_pct,
        )

        # Next state is terminal (all zeros)
        zero_state: dict[str, float] = {f: 0.0 for f in _FEATURE_NAMES}

        self.record_step(
            state=terminal_state,
            action=ACTION_EXIT,
            reward=reward,
            next_state=zero_state,
            done=True,
        )


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def build_rl_exit_agent(
    db_path: str,
    checkpoint_path: str = "models/rl_exit_agent.pkl",
    learning_rate: float = 0.1,
    discount: float = 0.95,
    epsilon: float = 0.1,
    warm_start: bool = True,
) -> RLExitAgent:
    """Create and optionally restore an ``RLExitAgent`` from disk.

    If a checkpoint exists at ``checkpoint_path`` it is loaded; otherwise the
    agent optionally warm-starts from closed trades in ``db_path``.

    Args:
        db_path: Path to ``trading_bot.db``.
        checkpoint_path: Where to look for (and save) the Q-table pickle.
        learning_rate: Alpha for Q-updates.
        discount: Gamma for future reward discounting.
        epsilon: Exploration rate.
        warm_start: If True and no checkpoint exists, call ``replay_from_db``.

    Returns:
        Configured ``RLExitAgent`` ready for use.
    """
    agent = RLExitAgent(
        db_path=db_path,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
    )

    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        try:
            agent.load(str(ckpt))
            logger.info("RLExitAgent checkpoint restored from {}", checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RLExitAgent checkpoint load failed ({}), starting fresh", exc
            )
            if warm_start:
                agent.replay_from_db()
    else:
        logger.info("No checkpoint at {} — fresh agent", checkpoint_path)
        if warm_start:
            agent.replay_from_db()

    return agent
