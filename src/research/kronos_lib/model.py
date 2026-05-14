# Adapter module — re-exports Kronos from kronos.py for clean import paths.
# Vendored from https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a
# MIT License — see LICENSE.kronos in this directory.

from .kronos import Kronos

__all__ = ["Kronos"]
