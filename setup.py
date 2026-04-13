#!/usr/bin/env python3
"""Setup and validation script for Trading Bot.

This script validates all configurations before starting the bot
to ensure safe and correct operation.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better formatting: pip install rich")


class SetupValidator:
    """Validates trading bot configuration before startup."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.errors = []
        self.warnings = []
        self.info = []
        self.base_dir = Path(__file__).parent.absolute()

    def print_header(self, text: str):
        """Print a formatted header."""
        if self.console:
            self.console.print(Panel(text, style="bold blue"))
        else:
            print(f"\n{'=' * 60}")
            print(text)
            print('=' * 60)

    def print_success(self, text: str):
        """Print a success message."""
        if self.console:
            self.console.print(f"✓ {text}", style="green")
        else:
            print(f"✓ {text}")

    def print_error(self, text: str):
        """Print an error message."""
        self.errors.append(text)
        if self.console:
            self.console.print(f"✗ {text}", style="red")
        else:
            print(f"✗ {text}")

    def print_warning(self, text: str):
        """Print a warning message."""
        self.warnings.append(text)
        if self.console:
            self.console.print(f"⚠ {text}", style="yellow")
        else:
            print(f"⚠ {text}")

    def print_info(self, text: str):
        """Print an info message."""
        self.info.append(text)
        if self.console:
            self.console.print(f"ℹ {text}", style="cyan")
        else:
            print(f"ℹ {text}")

    def check_env_file(self) -> bool:
        """Check if .env file exists and is readable."""
        self.print_header("Checking Environment Configuration")

        env_file = self.base_dir / ".env"
        env_example = self.base_dir / ".env.example"

        if not env_file.exists():
            self.print_error(".env file not found!")
            self.print_info("Copy .env.example to .env and fill in your credentials")

            if env_example.exists():
                self.print_info("Run: cp .env.example .env")
            return False

        self.print_success(".env file exists")

        # Check if .env has been modified from example
        with open(env_file) as f:
            env_content = f.read()

        if "your_upstox_api_key" in env_content or "your_" in env_content:
            self.print_error(".env file contains placeholder values!")
            self.print_info("Please replace all 'your_*' placeholders with actual values")
            return False

        return True

    def check_required_variables(self) -> dict:
        """Check required environment variables."""
        self.print_header("Checking Required Variables")

        required = {
            "UPSTOX_CLIENT_ID": "Upstox API Key",
            "UPSTOX_CLIENT_SECRET": "Upstox API Secret",
            "UPSTOX_REDIRECT_URI": "OAuth Redirect URI",
        }

        recommended = {
            "UPSTOX_USERNAME": "Username (optional, for auto-login)",
            "UPSTOX_PASSWORD": "Password (optional, for auto-login)",
            "UPSTOX_PIN_CODE": "PIN Code (optional, for auto-login)",
            "UPSTOX_TOTP_SECRET": "TOTP Secret (optional, for auto-login)",
        }

        env_vars = {}

        # Load from .env file
        env_file = self.base_dir / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        all_good = True

        # Check required
        for var, description in required.items():
            value = env_vars.get(var, os.getenv(var, ''))
            if not value or value.startswith('your_'):
                self.print_error(f"{var} ({description}) is missing or invalid")
                all_good = False
            else:
                masked = value[:4] + '****' if len(value) > 4 else '****'
                self.print_success(f"{var} is set ({masked})")

        # Check recommended
        for var, description in recommended.items():
            value = env_vars.get(var, os.getenv(var, ''))
            if not value:
                self.print_warning(f"{var} ({description}) not set - manual login required")
            else:
                self.print_success(f"{var} is set")

        return env_vars if all_good else {}

    def validate_trading_mode(self, env_vars: dict):
        """Validate trading mode is set to paper."""
        self.print_header("Checking Trading Mode")

        mode = env_vars.get('TRADING_MODE', 'paper').lower()

        if mode == 'paper':
            self.print_success("TRADING_MODE is set to 'paper' - Safe for testing")
        elif mode == 'live':
            self.print_error("TRADING_MODE is set to 'LIVE'!")
            self.print_error("This will place REAL orders with REAL money!")
            self.print_info("Set TRADING_MODE=paper for testing")
        else:
            self.print_warning(f"TRADING_MODE is '{mode}' - should be 'paper' or 'live'")

        return mode == 'paper'

    def validate_capital_settings(self, env_vars: dict):
        """Validate capital and risk settings."""
        self.print_header("Checking Capital & Risk Settings")

        try:
            capital = int(env_vars.get('CAPITAL', '0'))
            if capital <= 0:
                self.print_error("CAPITAL must be greater than 0")
                return False

            # Convert paisa to rupees for display
            capital_rupees = capital / 100
            self.print_success(f"CAPITAL: ₹{capital_rupees:,.2f}")

            max_daily_loss = int(env_vars.get('MAX_DAILY_LOSS', '0'))
            if max_daily_loss > 0:
                loss_pct = (max_daily_loss / capital) * 100
                self.print_success(f"MAX_DAILY_LOSS: ₹{max_daily_loss/100:,.2f} ({loss_pct:.1f}% of capital)")

                if loss_pct > 5:
                    self.print_warning(f"Daily loss limit is high ({loss_pct:.1f}%)")
                elif loss_pct < 1:
                    self.print_warning(f"Daily loss limit is very conservative ({loss_pct:.1f}%)")
            else:
                self.print_error("MAX_DAILY_LOSS not set")

            max_positions = int(env_vars.get('MAX_OPEN_POSITIONS', '0'))
            if max_positions > 0:
                self.print_success(f"MAX_OPEN_POSITIONS: {max_positions}")
            else:
                self.print_warning("MAX_OPEN_POSITIONS not set (default: 5)")

            return True

        except ValueError as e:
            self.print_error(f"Invalid capital settings: {e}")
            return False

    def check_directories(self) -> bool:
        """Check required directories exist."""
        self.print_header("Checking Directory Structure")

        required_dirs = ['data', 'logs', 'config/strategies']
        all_exist = True

        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                self.print_success(f"{dir_name}/ exists")
            else:
                self.print_error(f"{dir_name}/ does not exist")
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.print_info(f"Created {dir_name}/ directory")
                except Exception as e:
                    self.print_error(f"Failed to create {dir_name}/: {e}")
                    all_exist = False

        return all_exist

    def check_strategy_files(self) -> bool:
        """Check strategy configuration files."""
        self.print_header("Checking Strategy Configurations")

        strategies_dir = self.base_dir / "config" / "strategies"
        if not strategies_dir.exists():
            self.print_error("config/strategies/ directory not found")
            return False

        strategy_files = list(strategies_dir.glob("*.json"))

        if not strategy_files:
            self.print_error("No strategy JSON files found!")
            self.print_info("Create strategy files in config/strategies/")
            return False

        all_valid = True
        for strategy_file in strategy_files:
            try:
                with open(strategy_file) as f:
                    strategy = json.load(f)

                name = strategy.get('name', 'Unknown')
                enabled = strategy.get('enabled', False)

                if enabled:
                    self.print_success(f"{strategy_file.name}: '{name}' (enabled)")
                else:
                    self.print_warning(f"{strategy_file.name}: '{name}' (disabled)")

                # Validate required fields
                required_fields = ['name', 'entry_sets', 'exit_rules']
                for field in required_fields:
                    if field not in strategy:
                        self.print_error(f"  Missing required field: {field}")
                        all_valid = False

            except json.JSONDecodeError as e:
                self.print_error(f"{strategy_file.name}: Invalid JSON - {e}")
                all_valid = False
            except Exception as e:
                self.print_error(f"{strategy_file.name}: Error - {e}")
                all_valid = False

        return all_valid

    def check_database(self) -> bool:
        """Check database can be initialized."""
        self.print_header("Checking Database Connection")

        db_path = self.base_dir / "data" / "trading_bot.db"

        try:
            # Try importing and creating tables
            sys.path.insert(0, str(self.base_dir))
            from src.persistence.models import Base
            from sqlalchemy import create_engine

            engine = create_engine(f"sqlite:///{db_path}", echo=False)
            Base.metadata.create_all(engine)

            self.print_success(f"Database ready: {db_path}")
            return True

        except Exception as e:
            self.print_error(f"Database initialization failed: {e}")
            return False

    def check_python_dependencies(self) -> bool:
        """Check required Python packages."""
        self.print_header("Checking Python Dependencies")

        required = {
            'pydantic': 'Data validation',
            'pydantic_settings': 'Settings management',
            'sqlalchemy': 'Database ORM',
            'pandas': 'Data analysis',
            'loguru': 'Logging',
            'requests': 'HTTP client',
            'websocket-client': 'WebSocket client',
        }

        optional = {
            'rich': 'Formatted output',
            'pandas_ta': 'Technical indicators',
            'numpy': 'Numerical computing',
        }

        all_good = True

        for package, description in required.items():
            try:
                __import__(package.replace('-', '_'))
                self.print_success(f"{package} ({description})")
            except ImportError:
                self.print_error(f"{package} ({description}) - NOT INSTALLED")
                all_good = False

        for package, description in optional.items():
            try:
                __import__(package.replace('-', '_'))
                self.print_success(f"{package} ({description}) - optional")
            except ImportError:
                self.print_warning(f"{package} ({description}) - optional, not installed")

        return all_good

    def validate_upstox_credentials(self, env_vars: dict) -> bool:
        """Validate Upstox credential format."""
        self.print_header("Validating Upstox Credentials")

        client_id = env_vars.get('UPSTOX_CLIENT_ID', '')
        client_secret = env_vars.get('UPSTOX_CLIENT_SECRET', '')
        redirect_uri = env_vars.get('UPSTOX_REDIRECT_URI', '')

        valid = True

        # Check client ID format (typically alphanumeric)
        if len(client_id) < 10:
            self.print_warning("UPSTOX_CLIENT_ID seems short (should be ~30+ chars)")

        # Check client secret format
        if len(client_secret) < 10:
            self.print_warning("UPSTOX_CLIENT_SECRET seems short (should be ~40+ chars)")

        # Check redirect URI
        if not redirect_uri.startswith('http'):
            self.print_error("UPSTOX_REDIRECT_URI must be a valid URL")
            valid = False
        elif 'localhost' in redirect_uri or '127.0.0.1' in redirect_uri:
            self.print_success("UPSTOX_REDIRECT_URI is set to localhost (correct for testing)")

        return valid

    def print_summary(self):
        """Print validation summary."""
        self.print_header("Validation Summary")

        if self.console:
            table = Table(title="Results")
            table.add_column("Type", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_row("Errors", str(len(self.errors)))
            table.add_row("Warnings", str(len(self.warnings)))
            table.add_row("Info", str(len(self.info)))
            self.console.print(table)
        else:
            print(f"\nErrors: {len(self.errors)}")
            print(f"Warnings: {len(self.warnings)}")
            print(f"Info: {len(self.info)}")

        if self.errors:
            if self.console:
                self.console.print("\n[red]❌ VALIDATION FAILED - Fix errors before starting[/red]")
            else:
                print("\n❌ VALIDATION FAILED - Fix errors before starting")
            return False
        elif self.warnings:
            if self.console:
                self.console.print("\n[yellow]⚠ VALIDATION PASSED WITH WARNINGS - Review warnings[/yellow]")
            else:
                print("\n⚠ VALIDATION PASSED WITH WARNINGS - Review warnings")
            return True
        else:
            if self.console:
                self.console.print("\n[green]✅ ALL CHECKS PASSED - Ready to start![/green]")
            else:
                print("\n✅ ALL CHECKS PASSED - Ready to start!")
            return True

    def run(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 60)
        print("Trading Bot Setup Validator")
        print("=" * 60)

        # Check dependencies first
        if not self.check_python_dependencies():
            print("\nInstall missing dependencies:")
            print("pip install -r requirements.txt")
            return False

        # Check environment
        if not self.check_env_file():
            return False

        env_vars = self.check_required_variables()
        if not env_vars:
            return False

        # Validate specific settings
        is_paper = self.validate_trading_mode(env_vars)
        self.validate_capital_settings(env_vars)
        self.validate_upstox_credentials(env_vars)

        # Check infrastructure
        self.check_directories()
        self.check_strategy_files()
        self.check_database()

        # Print summary
        can_start = self.print_summary()

        if can_start and is_paper:
            print("\n" + "=" * 60)
            print("To start the bot:")
            print("  python -m src.main")
            print("=" * 60)
        elif can_start and not is_paper:
            print("\n" + "=" * 60)
            print("⚠️  WARNING: Trading mode is set to LIVE!")
            print("   This will place REAL orders with REAL money!")
            print("=" * 60)

        return can_start


def main():
    """Main entry point."""
    validator = SetupValidator()
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
