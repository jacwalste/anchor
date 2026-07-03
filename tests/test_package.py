"""Placeholder until real tests land with the claim schema; keeps pytest green."""

import anchor


def test_package_imports() -> None:
    assert anchor.__doc__ is not None
