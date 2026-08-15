"""Stub for `ssr_eval`, which A2SB imports but does not need for inference.

The real package depends on `mysql-python`, a Python 2-era library that cannot
build on a current toolchain, so it is not installable. A2SB only touches it in
validation paths (`ssr_eval.metrics.AudioMetrics`), never in `predict`.

This shim satisfies the import and nothing else. Anything that actually calls
into it raises, so a stub can never be mistaken for a working metric.
"""

from . import metrics

__all__ = ["metrics"]
