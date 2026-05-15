"""Offline evaluation scaffolds for the quantitative-trading agent.

The submodules here intentionally live outside ``src/quantitative_trading``
so they don't ship with the production package — they're calibration
loops that may pull in heavy or paid dependencies (real LLM calls,
Berkshire 13F parsing, etc.). Add new evals as fresh submodules and
wire them up via standalone ``python -m`` entry points.
"""
