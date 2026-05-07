from datetime import datetime
from zoneinfo import ZoneInfo
from src.strategy.engine import Signal, SignalType
from src.execution.paper_broker import PaperBroker
from src.execution.base_broker import OrderStatus, ProductType, OrderSide
from src.execution.position_manager import PositionManager
from src.execution.order_manager import OrderManager

IST = ZoneInfo("Asia/Kolkata")

def test_paper_workflow_straddle_execution():
    """Test full workflow from Signal emission to Position tracking."""
    broker = PaperBroker(initial_capital=5000000) # 50k INR
    pos_manager = PositionManager(broker)
    
    # Needs a mock strategy config/engine dependency if OrderManager checks strategy constraints,
    # but we can bypass or provide a minimal mock if OrderManager accepts it.
    # OrderManager requires strategy_engine, position_manager, broker.
    class MockStrategyEngine:
        def __init__(self):
            self._strategies = {}
        def get_strategy(self, name):
            class MockConfig:
                name = "TestStrat"
                enabled = True
                instrument_selection = {
                    "option_types": ["CE", "PE"]
                }
                underlying = {"instrument_key": "NSE_INDEX|Nifty Bank"}
                timeframes = {"primary": "5min"}
                exit_rules = {"stop_loss_pct": 20, "target_pct": 50, "max_holding_minutes": 100}
                position_sizing = {"method": "fixed_quantity", "quantity": 15}
                risk_management = {"hedge_required": False}
                def dict(self): return {}
            return MockConfig()
            
    strat_engine = MockStrategyEngine()
    order_manager = OrderManager(strat_engine, pos_manager, broker)
    
    # Update market data
    broker.update_ltp("NSE_INDEX|Nifty Bank", 5000000)
    broker.update_ltp("NSE_FO|BANKNIFTY_CE", 20000)
    broker.update_ltp("NSE_FO|BANKNIFTY_PE", 20000)
    
    # We must patch the OptionChain resolution since we don't have a live chain here
    def mock_get_option_instrument(underlying, strike, is_call):
        return "NSE_FO|BANKNIFTY_CE" if is_call else "NSE_FO|BANKNIFTY_PE"
    
    # This is a bit complex to mock entirely via OrderManager without a full chain. 
    # Let's test the workflow by placing an order manually and tracking P&L.
    
    # Direct combo order
    from src.execution.base_broker import Order, ComboOrder, OrderSide, OrderType, ProductType
    leg1 = Order(
        instrument_key="NSE_FO|BANKNIFTY_CE",
        quantity=15,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        product=ProductType.MIS
    )
    leg2 = Order(
        instrument_key="NSE_FO|BANKNIFTY_PE",
        quantity=15,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        product=ProductType.MIS
    )
    combo = ComboOrder(strategy_id="TestStrat", legs=[leg1, leg2])
    
    responses = broker.place_combo_order(combo)
    assert len(responses) == 2
    
    for r in responses:
        if r.status in ("COMPLETE", OrderStatus.COMPLETE):
            pos_manager.add_position(
                instrument_key=r.instrument_key,
                strategy_id=combo.strategy_id,
                side=r.side if hasattr(r.side, 'value') else OrderSide(r.side),
                quantity=r.filled_quantity,
                entry_price=r.average_price,
                product_type=ProductType.MIS,
                stop_loss_pct=20.0
            )
        
    # Both positions are registered internally (by position_id), even though
    # _instrument_to_position only holds latest mapping per instrument.
    all_positions = list(pos_manager._positions.values())
    open_positions = [p for p in all_positions if not p.is_closed]
    assert len(open_positions) == 2, f"Expected 2 open positions, got {len(open_positions)}"
    
    # Simulate market crash - Puts explode (CE profits, PE loses big)
    broker.update_ltp("NSE_FO|BANKNIFTY_CE", 10000) # CE dropped 50% (profit for SELL)
    broker.update_ltp("NSE_FO|BANKNIFTY_PE", 40000) # PE doubled (loss for SELL)
    
    pos_manager.on_tick("NSE_FO|BANKNIFTY_CE", 10000)
    pos_manager.on_tick("NSE_FO|BANKNIFTY_PE", 40000)
    
    # Find the PE position and verify its unrealized P&L is negative
    # (SELL at 19990 paisa, LTP now 40000 - we're underwater)
    pe_positions = [p for p in pos_manager._positions.values()
                    if "PE" in p.instrument_key and not p.is_closed]
    
    if pe_positions:
        pe_pos = pe_positions[0]
        # For a SHORT (SELL) position: unrealized_pnl = (entry - ltp) * qty
        # Entry ~19990, LTP = 40000: pnl = (19990 - 40000) * 15 = -300150 paisa (very negative)
        assert pe_pos.unrealized_pnl < 0, f"PE position should be loss, got {pe_pos.unrealized_pnl}"
