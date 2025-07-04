# -*- coding: utf-8 -*-

"""Tests for chemcaption.featurize.miscellaneous subpackage."""

import re

import numpy as np
import pytest

from chemcaption.featurize.miscellaneous import SVGFeaturizer
from tests.conftests import DISPATCH_MAP, PROPERTY_BANK, get_molecules

KIND = "selfies"
MOLECULE = DISPATCH_MAP[KIND]

# Implemented tests for miscellaneous featurizers.

__all__ = [
    "test_svg_featurizer",
]


"""Test for molecule-to-SVG featurizer."""

SVG_R = r"(?:<\?xml\b[^>]*>[^<]*)?(?:<!--.*?-->[^<]*)*(?:<svg|<!DOCTYPE svg)\b"
SVG_RE = re.compile(SVG_R, re.DOTALL)


@pytest.mark.parametrize(
    "test_input",
    get_molecules(property_bank=PROPERTY_BANK, representation_name=KIND),
)
def test_svg_featurizer(test_input):
    """Test SVGFeaturizer."""
    featurizer = SVGFeaturizer()
    molecule = MOLECULE(test_input)

    results = featurizer.featurize(molecule)

    assert SVG_RE.match(results)
