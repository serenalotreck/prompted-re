"""
Spot checks for the functions in evaluation_utils.py

Author: Serena G. Lotreck
"""
import pytest
import pandas as pd
import sys
sys.path.append ('../')
import evaluation_utils as eu


@pytest.fixture
def basic_preds():
    return pd.DataFrame(
        {'E1': ['hello', 'Sparty', 'Sparty', 'cool', 'computer'],
        'R': ['REL', 'REL', 'REL', 'REL', 'REL'],
        'E2': ['world', 'goodbye', 'ice cream', 'MSU', 'nice']}
        )

@pytest.fixture
def basic_gold():
    return pd.DataFrame(
        {'E1': ['hello', 'goodbye', 'I', 'Sparty', 'MSU', 'computer'],
        'R': ['REL', 'REL', 'REL', 'REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'green', 'ice cream', 'cool', 'annoying']}
        )