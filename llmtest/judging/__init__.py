"""Judging pipeline (TESTPLAN 6.1/6.2) -- cohort packets, adapters, runner, aggregation.

Phase-separated from battery execution: batteries only emit needs_judging
rows; everything here reads those rows back and turns them into blinded
judge packets, judgments, and (at table time, in aggregate.py) scores.
"""
from __future__ import annotations
