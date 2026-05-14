#!/bin/bash
# Trading bot lifecycle script for scheduled agents

cd /home/workspace/Trading

# Check if it's a trading day
python3 -c "
from datetime import date
from src.utils.holidays import is_trading_day

if is_trading_day(date.today()):
    exit(0)  # It's a trading day
else:
    exit(1)  # Not a trading day
" || exit 0  # Exit silently if not a trading day

# Run the bot (this will be the START agent)
if [ "$1" = "start" ]; then
    echo "Starting trading bot..."
    python3 -m src.main &>> logs/bot_$(date +%Y%m%d).log &
    echo $! > logs/bot.pid
    echo "Bot started with PID $(cat logs/bot.pid)"

# Stop the bot (this will be the STOP agent)
elif [ "$1" = "stop" ]; then
    echo "Stopping trading bot..."
    if [ -f logs/bot.pid ]; then
        kill $(cat logs/bot.pid) 2>/dev/null
        rm logs/bot.pid
        echo "Bot stopped"
    fi
fi
