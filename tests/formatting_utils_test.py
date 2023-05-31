"""
Spot checks for the functions in formatting_utils.py

Author: Serena G. Lotreck
"""
import pytest
import jsonlines
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
sys.path.append('../prompted_re')
import formatting_utils as fu

################################ Unit Tests ###################################

########### format_dygiepp_df ##############


@pytest.fixture
def dygiepp_file(tmp_path_factory):
    dygp = [{
        'doc_key':
        'doc1',
        'sentences': [['hello', 'world', 'I', 'am', 'Sparty', '.'],
                      ['I', 'work', 'at', 'MSU', '.']],
        'relations': [[[2, 2, 4, 4, "is"]], [[6, 6, 9, 9, "works-at"]]],
        'predicted_relations': [[[2, 2, 4, 4, "is"]], []]
    }, {
        'doc_key':
        'doc2',
        'sentences': [[
            'MSU', 'has', 'two', 'colors', 'which', 'are', 'green', 'and',
            'white', '.'
        ],
                      [
                          'MSUs', 'slogans', 'are', 'Spartans', 'Will', 'and',
                          'Go', 'Green', ',', 'Go', 'White', '.'
                      ]],
        'relations': [[[0, 0, 6, 8, "has-colors"]],
                      [[10, 10, 13, 14, "has-slogan"],
                       [10, 10, 16, 20, "has-slogan"]]],
        'predicted_relations': [[[0, 0, 6, 8, "has-colors"]],
                                [[10, 10, 13, 14, "has-slogan"]]]
    }]
    fn = tmp_path_factory.mktemp("dygiepp_files") / "preds_and_gold.jsonl"
    with jsonlines.open(fn, 'w') as writer:
        writer.write_all(dygp)

    return fn


@pytest.fixture
def filter_type():
    return ["is", "works-at", "has-slogan"]


@pytest.fixture
def dygiepp_df():
    dset_dict = {
        'doc_key': ['doc1', 'doc2'],
        'trips': [[['I', 'is', 'Sparty'], ['I', 'works-at', 'MSU']],
                  [['MSUs', 'has-slogan', 'Spartans Will'],
                   ['MSUs', 'has-slogan', 'Go Green , Go White']]],
        'preds': [[['I', 'is', 'Sparty']],
                  [['MSUs', 'has-slogan', 'Spartans Will']]]
    }
    dset_df = pd.DataFrame(dset_dict)

    return dset_df


def test_format_dygeipp_df(dygiepp_file, filter_type, dygiepp_df):

    df, dropped_gold, dropped_pred = fu.format_dygiepp_df(dygiepp_file, filter_type)

    print(df.to_string())
    print(dygiepp_df.to_string())
    print(df.dtypes)
    print(dygiepp_df.dtypes)

    assert_frame_equal(df, dygiepp_df)
    assert dropped_gold == 1
    assert dropped_pred == 1
