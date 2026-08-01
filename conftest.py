"""Test configuration: import path and Hypothesis profiles."""

import sys
from pathlib import Path

from hypothesis import HealthCheck, settings

sys.path.insert(0, str(Path(__file__).parent))

# Default: fast enough to run on every change.
settings.register_profile(
    "default",
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Deep: for the sweep run before tagging a release or submitting the paper.
#   pytest --hypothesis-profile=deep
settings.register_profile(
    "deep",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

settings.load_profile("default")
