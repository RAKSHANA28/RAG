#!/usr/bin/env python3
"""
AQU-MR: Root Entry Point
Delegates commands to aqu_mr/run.py
"""
import os
import sys

# Ensure aqu_mr is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aqu_mr"))

from aqu_mr.run import main

if __name__ == "__main__":
    main()
