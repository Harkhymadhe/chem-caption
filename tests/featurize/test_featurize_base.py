# -*- coding: utf-8 -*-

"""Unit tests for chemcaption.featurize.base submodule."""

from chemcaption.featurize.base import Comparator, MultipleComparator, MultipleFeaturizer
from chemcaption.featurize.comparator import AtomCountComparator, IsomerismComparator
from chemcaption.featurize.electronicity import HydrogenAcceptorCountFeaturizer
from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from chemcaption.molecules import SMILESMolecule

__all__ = ["test_multiple_featurizer", "test_multiple_comparator", "test_comparator"]


def test_multiple_featurizer():
    """Tests the MultipleFeaturizer."""
    smiles = SMILESMolecule("CCCC")

    featurizer = MultipleFeaturizer(
        featurizers=[
            HydrogenAcceptorCountFeaturizer(),
            ChiralCenterCountFeaturizer(),
        ]
    )

    results = featurizer.featurize(smiles)
    assert len(results[0]) == 2
    assert len(results[0]) == len(featurizer.feature_labels)

    text = featurizer.text_featurize(pos_key="noun", molecule=smiles)
    assert len(text) == len(featurizer.featurizers)

    smiles_list = [SMILESMolecule("CCCC"), SMILESMolecule("O")]

    results = featurizer.text_featurize_many(molecules=smiles_list)

    assert len(results) == len(smiles_list)


def test_multiple_comparator():
    """Test the MultipleComparator."""

    molecules = [
        SMILESMolecule("[C-]#[O+]"),  # Carbon II Oxide
        SMILESMolecule("N#N"),  # Nitrogen molecule
        SMILESMolecule("N#[O+]"),  # Nitrous Ion
        SMILESMolecule("[C-]#N"),  # Cyanide Ion
    ]

    comparator = MultipleComparator(
        comparators=[
            AtomCountComparator(),
            IsomerismComparator(),
        ]
    )

    results = comparator.compare(molecules)

    assert len(results[0]) == len(comparator.comparators)

    implementors = comparator.implementors()

    assert isinstance(implementors, list)

    assert len(comparator.comparators) == len(comparator.feature_labels)

    assert (comparator.featurize(molecules) == results).all()

    comparator = comparator.fit_on_comparators()

    assert comparator.comparators is None

    comparator = comparator.fit_on_comparators([AtomCountComparator()])

    assert len(comparator.comparators) == 1


def test_comparator():
    """Test the Comparator."""

    molecules = [
        SMILESMolecule("[C-]#[O+]"),  # Carbon II Oxide
        SMILESMolecule("N#N"),  # Nitrogen molecule
        SMILESMolecule("N#[O+]"),  # Nitrous Ion
        SMILESMolecule("[C-]#N"),  # Cyanide Ion
    ]

    comparator = Comparator(
        featurizers=[
            HydrogenAcceptorCountFeaturizer(),
            ChiralCenterCountFeaturizer(),
        ]
    )

    results = comparator.compare(molecules)

    assert len(results[0]) == len(comparator.featurizers)

    implementors = comparator.implementors()

    assert isinstance(implementors, list)

    assert len(comparator.featurizers) == len(comparator.feature_labels)

    assert (comparator.featurize(molecules) == results).all()

    comparator = comparator.fit_on_featurizers()

    assert comparator.featurizers is None

    comparator = comparator.fit_on_featurizers([HydrogenAcceptorCountFeaturizer()])

    assert len(comparator.featurizers) == 1
