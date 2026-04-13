"""Tests that LearningIntegration.backtest_mode skips lesson persistence."""
from unittest.mock import MagicMock, patch

from src.learning.integration import LearningIntegration


def test_backtest_mode_skips_lesson_persistence():
    """In backtest_mode, save_lesson must NOT be called."""
    mock_session = MagicMock()

    with patch("src.learning.integration.LearningEngine") as MockEngine, \
         patch("src.learning.integration.LearningPersistence") as MockPersistence:

        mock_engine_instance = MockEngine.return_value
        mock_persistence_instance = MockPersistence.return_value

        # Simulate a losing trade (triggers save_lesson in live mode)
        mock_engine_instance.process_trade.return_value = {
            "was_win": False,
            "pnl_rupees": -500.0,
        }
        mock_engine_instance._analyzer.get_lessons.return_value = []
        mock_engine_instance._analyzer.get_strategy_performance.return_value = None

        integration = LearningIntegration(
            db_session=mock_session,
            backtest_mode=True,
        )

        # Create mock position
        mock_position = MagicMock()
        mock_position.position_id = "POS_001"
        mock_position.strategy_id = "test_strategy"
        mock_position.entry_price = 500000
        mock_position.exit_price = 490000
        mock_position.quantity = 1
        mock_position.side = MagicMock()
        mock_position.side.value = "BUY"
        mock_position.entry_time = MagicMock()
        mock_position.exit_time = MagicMock()
        mock_position.is_closed = True
        mock_position.total_realized_pnl = -10000

        integration.on_position_closed(mock_position)

        # In backtest mode, save_lesson must NOT be called
        mock_persistence_instance.save_lesson.assert_not_called()


def test_live_mode_saves_lesson_on_loss():
    """In live mode (default), save_lesson IS called for losing trades."""
    mock_session = MagicMock()

    with patch("src.learning.integration.LearningEngine") as MockEngine, \
         patch("src.learning.integration.LearningPersistence") as MockPersistence:

        mock_engine_instance = MockEngine.return_value
        mock_persistence_instance = MockPersistence.return_value

        # Build a fake lesson that has not been applied
        fake_lesson = MagicMock()
        fake_lesson.applied = False

        mock_engine_instance.process_trade.return_value = {
            "was_win": False,
            "pnl_rupees": -500.0,
        }
        mock_engine_instance._analyzer.get_lessons.return_value = [fake_lesson]
        mock_engine_instance._analyzer.get_strategy_performance.return_value = None

        # Default backtest_mode=False → live mode
        integration = LearningIntegration(
            db_session=mock_session,
            backtest_mode=False,
        )

        mock_position = MagicMock()
        mock_position.position_id = "POS_002"
        mock_position.strategy_id = "test_strategy"
        mock_position.entry_price = 500000
        mock_position.exit_price = 490000
        mock_position.quantity = 1
        mock_position.side = MagicMock()
        mock_position.side.value = "BUY"
        mock_position.entry_time = MagicMock()
        mock_position.exit_time = MagicMock()
        mock_position.is_closed = True
        mock_position.total_realized_pnl = -10000

        integration.on_position_closed(mock_position)

        # In live mode, save_lesson MUST be called for the losing trade
        mock_persistence_instance.save_lesson.assert_called_once_with(fake_lesson)
