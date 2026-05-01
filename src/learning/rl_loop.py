"""Continuous Reinforcement Learning Loop.

Runs at 3:30 PM (15:30 IST) to analyze the day's trades, extract lessons,
and dynamically tune the strategy JSON configuration files for the next
trading session. Enforces strict safety bounds to prevent infinite risk.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.agents.base_agent import BaseAgent, extract_json
from src.agents.llm_client import LLMClient
from src.learning.trade_analyzer import TradeAnalyzer

SYSTEM_PROMPT = """You are the AI Risk Manager and Strategy Optimizer.
Your job is to review today's trading performance and suggest parameter updates for the strategy JSON configs to improve performance tomorrow.

IMPORTANT: Output ONLY the raw JSON object below. No reasoning, no explanation, no markdown fences, no text before or after the JSON.

{
    "strategy_name": "<name>",
    "suggested_updates": {
        "stop_loss_atr_multiplier": <float>,
        "quantity_multiplier": <float>
    },
    "day_of_week_adjustments": {
        "Monday": <0.5 to 1.5>,
        "Tuesday": <0.5 to 1.5>,
        "Wednesday": <0.5 to 1.5>,
        "Thursday": <0.5 to 1.5>,
        "Friday": <0.5 to 1.5>
    },
    "reasoning": "<one short sentence>"
}

Rules:
- If the strategy suffered whipsaws (small losses but correct direction), widen the stop_loss_atr_multiplier slightly.
- If the strategy was highly profitable with high win rate, increase the quantity_multiplier slightly (max 1.5).
- If the strategy suffered massive consecutive losses, reduce the quantity_multiplier (e.g. 0.5).
- If Monday consistently loses, set Monday adjustment to 0.5 or 0.75.
- If Tuesday/Wednesday consistently win, set them to 1.25.
- If Friday has mixed results due to overnight risk, keep at 0.75.
- Output updates ONLY if necessary. If no updates needed, output empty suggested_updates.
"""

class ContinuousRLLoop(BaseAgent):
    """Reinforcement learning agent that tweaks strategy JSON configs.
    
    Args:
        llm_client: Configured LLMClient instance.
        trade_analyzer: Initialized TradeAnalyzer containing today's performance.
    """

    def __init__(self, llm_client: LLMClient, trade_analyzer: TradeAnalyzer) -> None:
        super().__init__(llm_client)
        self._trade_analyzer = trade_analyzer

    @property
    def agent_name(self) -> str:
        return "rl_optimizer"

    def _build_prompt(self, context: dict[str, Any]) -> str:
        perf = context.get("performance", {})
        lessons = context.get("lessons", [])
        dow_stats = context.get("day_of_week_stats", {})
        
        prompt = (
            f"Analyze performance for Strategy: {perf.get('strategy_name', 'Unknown')}\n"
            f"Win Rate: {perf.get('win_rate', 0):.1%}\n"
            f"Total PnL (paisa): {perf.get('total_pnl_paisa', 0)}\n"
            f"Profit Factor: {perf.get('profit_factor', 0):.2f}\n"
            f"Max Consecutive Losses: {perf.get('max_consecutive_losses', 0)}\n\n"
        )
        
        if dow_stats:
            prompt += "--- DAY-OF-WEEK PERFORMANCE ---\n"
            for day, stats in dow_stats.items():
                prompt += f"{day}: {stats.get('trades', 0)} trades, Win Rate {stats.get('win_rate', 0):.0%}, PnL {stats.get('pnl', 0)} paisa\n"
            prompt += "\n"
            
        if lessons:
            prompt += "Today's Lessons:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson.description} (Impact: {lesson.impact})\n"
                
        return prompt

    def _build_system_prompt(self, lessons: list[str]) -> str:
        return SYSTEM_PROMPT

    def _parse_response(self, raw: str) -> dict[str, Any]:
        return extract_json(raw)

    def _fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fallback to no changes."""
        return {
            "strategy_name": context.get("performance", {}).get("strategy_name", "Unknown"),
            "suggested_updates": {},
            "reasoning": "Fallback: LLM unavailable, no changes made.",
        }
        
    def execute_daily_optimization(self, strategies_dir: Path) -> None:
        """Run the optimization loop for all tracked strategies and apply bounded updates."""
        all_perf = self._trade_analyzer.get_all_performance()
        if not all_perf:
            logger.info("RL Loop: No trades today to optimize.")
            return
            
        for strategy_name, perf in all_perf.items():
            if perf.total_trades < 3:
                continue # Need minimum sample size
                
            lessons = self._trade_analyzer.get_lessons(strategy=strategy_name)
            context = {
                "performance": perf.get_summary(),
                "lessons": lessons,
            }
            
            # 1. Ask LLM for parameter tweaks
            result = self.run(context=context)
            updates = result.get("suggested_updates", {})
            
            if not updates:
                continue
                
            logger.info("RL Loop suggested updates for {}: {} | Reason: {}", strategy_name, updates, result.get("reasoning", ""))
            
            # 2. Safety Bounds Enforcement
            safe_updates = {}
            if "stop_loss_atr_multiplier" in updates:
                val = float(updates["stop_loss_atr_multiplier"])
                # Hard limit: Max 2.5x ATR
                safe_updates["stop_loss_atr_multiplier"] = max(0.5, min(2.5, val))
                
            if "quantity_multiplier" in updates:
                val = float(updates["quantity_multiplier"])
                # Hard limit: Max 3.0x size, min 0.1x
                safe_updates["quantity_multiplier"] = max(0.1, min(3.0, val))
                
            if "risk_per_trade_percent" in updates:
                val = float(updates["risk_per_trade_percent"])
                # Hard limit: Max 2.0% risk per trade
                safe_updates["risk_per_trade_percent"] = max(0.1, min(2.0, val))
                
            # 3. Apply updates to JSON
            config_path = strategies_dir / f"{strategy_name}.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        
                    # Inject updates into parameters
                    params = config.get("parameters", {})
                    for k, v in safe_updates.items():
                        params[k] = v
                    config["parameters"] = params
                    
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=4)
                        
                    logger.info("RL Loop successfully applied bounds-checked updates to {}", config_path.name)
                except Exception as e:
                    logger.error("Failed to update config {}: {}", config_path.name, e)
            
            # Apply day-of-week adjustments (Scheme 7)
            dow_adjustments = result.get("day_of_week_adjustments", {})
            if dow_adjustments:
                # Store in a separate JSON file for the playbook to read
                dow_path = strategies_dir / f"{strategy_name}_dow_sizing.json"
                safe_dow = {}
                for day, val in dow_adjustments.items():
                    try:
                        safe_dow[day] = max(0.25, min(1.5, float(val)))
                    except (ValueError, TypeError):
                        safe_dow[day] = 1.0
                try:
                    with open(dow_path, "w") as f:
                        json.dump(safe_dow, f, indent=4)
                    logger.info("RL Loop wrote day-of-week sizing to {}", dow_path.name)
                except Exception as e:
                    logger.error("Failed to write DOW sizing {}: {}", dow_path.name, e)
