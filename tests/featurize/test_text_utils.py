# -*- coding: utf-8 -*-

"""Unit tests for `chemcaption.featurize.text_utils` submodule."""

import numpy as np

from chemcaption.featurize.text_utils import generate_info, generate_template, inspect_template

__all__ = [
    "test_text_utils",
]


def test_text_utils():
    """Test the text utils functions."""

    template = generate_template()

    assert isinstance(template, str)
    assert len(template) > 0

    # Use seeds to get both cases
    np.random.seed(42)

    inspected = inspect_template(template)

    assert inspected != template

    np.random.seed(3)

    inspected = inspect_template(template)

    assert inspected == template

    info = generate_info("single")

    assert isinstance(info, dict)
