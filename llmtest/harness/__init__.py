"""Harness-independent core (Part 2 Phase 1): normalized Trace schema +
HarnessAdapter ABC for running models through real agent harnesses (B8).

This package is intentionally standalone -- it does not import from or
modify any existing battery/schema/store module. The B8 row-schema wiring
is a later, deferred task.
"""
from __future__ import annotations
