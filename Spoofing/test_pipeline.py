"""
Test Spoofing Pipeline (placeholder)

This module is intentionally left as a template. The test pipeline
follows different rules from train / valid (for example, different
splits of engines, deterministic seeds, no overlapping windows on
the source side, and possibly different per-segment quotas).

When implementing it, you can import the shared helpers from
common_utils:

    from .common_utils import (
        ManifestWriter,
        create_empty_manifest,
        save_padded,
        derive_age_direction,
        safe_val,
        safe_str,
        TARGET_SR,
        TARGET_DURATION,
    )

and define a separate run_test_setting(...) function here.
"""


def run_test_setting(*args, **kwargs):
    """Placeholder. Implement test-specific logic here."""
    raise NotImplementedError(
        "Test pipeline is not implemented yet. "
        "Add your test-specific assignment and execution logic in this file."
    )
