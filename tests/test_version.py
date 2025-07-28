# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.version submodule."""

from chemcaption.version import get_git_hash, get_version

__all__ = ["test_version"]


def test_version():
    """Test the version functions"""

    version = get_version()

    assert isinstance(version, str)
    assert len(version) > 0

    hsh = get_git_hash()

    assert isinstance(hsh, str)
    assert len(hsh) > 0
    assert hsh != "UNHASHED"
