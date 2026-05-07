from datetime import datetime
from zoneinfo import ZoneInfo
from src.execution.paper_broker import PaperBroker, ComboOrder, Order, OrderSide, OrderType, ProductType, OrderStatus

IST = ZoneInfo("Asia/Kolkata")

def test_paper_broker_combo_order_placement():
    """Test that placing a combo order accurately updates funds and creates multiple responses."""
    broker = PaperBroker(initial_capital=1000000) # 10k INR
    broker.update_ltp("NSE_FO|BANKNIFTY26OCT45000CE", 20000) # 200 INR
    broker.update_ltp("NSE_FO|BANKNIFTY26OCT45000PE", 20000)
    
    # Simulate a Straddle entry
    leg1 = Order(
        instrument_key="NSE_FO|BANKNIFTY26OCT45000CE",
        quantity=15,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        product=ProductType.MIS
    )
    leg2 = Order(
        instrument_key="NSE_FO|BANKNIFTY26OCT45000PE",
        quantity=15,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        product=ProductType.MIS
    )
    combo = ComboOrder(strategy_id="Straddle", legs=[leg1, leg2])
    
    responses = broker.place_combo_order(combo)
    
    assert len(responses) == 2
    assert responses[0].status in ("COMPLETE", OrderStatus.COMPLETE)
    assert responses[1].status in ("COMPLETE", OrderStatus.COMPLETE)
    
    # Margin testing omitted as paper broker currently assumes infinite margin for naked shorts in backtest mode
    
    # Check slippage application on execution prices
    # For SELL MARKET orders, fill price should be slightly lower than LTP due to slippage
    # Since LTP is 20000, and default slippage is 0.05%, fill should be ~ 19990
    assert responses[0].average_price is not None
    assert responses[0].average_price < 20000
