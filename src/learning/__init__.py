"""Self-learning system for trade analysis and continuous improvement.

This module provides functionality to:
- Analyze trade outcomes and identify patterns
- Extract lessons from winning and losing trades
- Adjust strategy parameters based on performance
- Dynamically modify position sizing based on recent results
- Generate reports on lessons learned

Example:
    from src.learning import LearningEngine, LessonLearned

    engine = LearningEngine(db_session)
    engine.process_trade(trade_record)
    recommendations = engine.generate_recommendations()
"""

from src.learning.integration import (
    LearningIntegration,
    create_learning_integration_from_config,
)
from src.learning.persistence import LearningPersistence
from src.learning.trade_analyzer import (
    LearningEngine,
    LessonLearned,
    StrategyPerformance,
    TradeAnalyzer,
    create_learning_report,
)

__all__ = [
    "LearningEngine",
    "LessonLearned",
    "StrategyPerformance",
    "TradeAnalyzer",
    "create_learning_report",
    "LearningPersistence",
    "LearningIntegration",
    "create_learning_integration_from_config",
]
