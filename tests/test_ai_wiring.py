import sys
import os
import logging
from datetime import datetime

# Configure path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk.circuit_breaker import CircuitBreaker
from src.execution.position_manager import PositionManager
from src.execution.order_manager import OrderManager
from src.execution.paper_broker import PaperBroker
from src.strategy.engine import Signal, SignalType
from src.agents.pipeline import AgentPipeline
from src.agents.market_maker import MarketMakerAgent

def run_tests():
    logging.basicConfig(level=logging.DEBUG)
    
    print("\n=== TESTING SCHEME 4: CIRCUIT BREAKER DETOX ===")
    try:
        # Mocking the callback
        detox_called = [False]
        def mock_detox():
            detox_called[0] = True
            print("✅ Detox callback fired!")
            
        cb = CircuitBreaker(max_consecutive_losses=2, pause_minutes=30)
        cb.set_on_halt_callback(mock_detox)
        
        trade_details = {
            "direction": "BUY",
            "strategy": "TEST_STRATEGY",
            "sl_distance_pct": 0.5,
            "holding_time_min": 10,
            "hour": 10
        }
        print("Simulating consecutive losses...")
        cb.record_loss(trade_details)
        cb.record_loss(trade_details)
        
        print(f"Circuit breaker halted: {cb.is_halted()}")
        if detox_called[0]:
            print("✅ Circuit breaker detox callback triggered successfully")
        else:
            print("❌ Detox callback was NOT fired")
    except Exception as e:
        print(f"❌ Circuit breaker detox failed: {e}")

    print("\n=== TESTING SCHEME 1, 2 & 3: MORNING PLAYBOOK ===")
    try:
        # Bypassing the need for API keys by providing a mock LLM or just letting it fail gracefully
        print("Initializing MarketMakerAgent with dummy API key...")
        os.environ['GROQ_API_KEY'] = 'dummy_key'
        mm = MarketMakerAgent()
        print("✅ MarketMakerAgent initialized")
    except Exception as e:
        print(f"❌ Morning playbook failed: {e}")

    print("\n=== TESTING SCHEME 5, 6 & 7: SIGNAL VALIDATOR DEFAULT CONFIDENCE ===")
    try:
        sig = Signal(
            strategy_name="ORB_Breakout",
            set_name="Long_Set",
            instrument_key="NSE_INDEX|Nifty Bank",
            signal_type=SignalType.BUY,
            quantity=15, 
            price=5000000,
            stop_loss=4990000,
            target=5020000
        )
        print(f"Default confidence assigned to Signal: {sig.confidence}")
        if sig.confidence == 0.8:
            print("✅ Signal confidence default successfully set to 0.8")
        else:
            print("❌ Signal confidence is not 0.8")
            
        print("Signal dict representation:", sig.to_dict()["confidence"])
    except Exception as e:
        print(f"❌ Signal validation test failed: {e}")

if __name__ == "__main__":
    run_tests()
