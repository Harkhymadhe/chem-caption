# -*- coding: utf-8 -*-

"""Script to append new feature to the database."""

import os
from typing import List

import pandas as pd
from tqdm import tqdm

from chemcaption.featurize.base import AbstractFeaturizer
from chemcaption.featurize.miscellaneous import SVGFeaturizer
from chemcaption.featurize.spatial import PMIFeaturizer
from chemcaption.featurize.symmetry import PointGroupFeaturizer
from chemcaption.molecules import SMILESMolecule

BASE_DIR = os.getcwd()

MOLECULAR_BANK = pd.read_json(os.path.join(BASE_DIR, "molecular_bank.json"), orient="index")


def extend_dataset(
    smiles_list: List[str], dataset: pd.DataFrame, featurizer: AbstractFeaturizer
) -> pd.DataFrame:
    """Extends the dataset

    Args:
        smiles (List[int]): List of smiles molecules
        dataset (obj`pd.DataFrame`): current dataset.
        featurizer (obj`AbstractFeaturizer`): featurizer.

    Returns:
        pd.DataFrame: new merged dataset.
    """

    # Check if features already exist
    labels = featurizer.feature_labels

    if len(list(set(labels) & set(dataset.columns))) > 0:
        return dataset

    new_data = []

    for string in tqdm(smiles_list):
        s = [string] + featurizer.featurize(molecule=SMILESMolecule(string)).flatten().tolist()
        new_data.append(s)

    print("New data generated!")
    new_data = pd.DataFrame(data=new_data, columns=["smiles"] + featurizer.feature_labels)
    new_data = pd.merge(left=dataset, right=new_data, left_on="smiles", right_on="smiles")
    print("New data merged and persisted!")
    return new_data


if __name__ == "__main__":
    NEW_PATH = os.path.join(BASE_DIR.replace("legacy", ""), "merged_pubchem_response.csv")
    PROPERTY_BANK = pd.read_csv(NEW_PATH)
    smiles_list = PROPERTY_BANK["smiles"]

    print("Adding features.")

    featurizers = [PMIFeaturizer(), SVGFeaturizer(), PointGroupFeaturizer()]

    for featurizer in featurizers:
        PROPERTY_BANK = extend_dataset(smiles_list, PROPERTY_BANK, featurizer)

    PROPERTY_BANK.to_csv(NEW_PATH, index=False)

    print("Database updated.")
