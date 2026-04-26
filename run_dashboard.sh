#!/bin/bash
cd /Users/sandeepvangapandu/Downloads/Trading || exit 1
exec python3 -m streamlit run src/dashboard/dashboard.py --server.headless true --server.port "${PORT:-8501}"
