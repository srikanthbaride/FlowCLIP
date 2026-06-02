"""Shared pytest fixtures / path setup for the FlowCLIP test suite.

Adds the repository root to ``sys.path`` so ``modules`` and the config
files resolve regardless of the directory pytest is invoked from.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
