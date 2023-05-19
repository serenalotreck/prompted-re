"""
Spot checks for the functions in evaluation_utils.py

Author: Serena G. Lotreck
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
sys.path.append ('../')
import evaluation_utils as eu


################################# Unit Tests ##################################

                              ##### symmetric_drop #####

@pytest.fixture
def yes_drop_input():
    return pd.DataFrame(
        {'E1': ['E1', 'E2', 'E3', 'E1'],
        'R': ['R1', 'R1', 'R2', 'R1'],
        'E2': ['E2', 'E1', 'E1', 'E3']}
        )

@pytest.fixture
def no_drop_input():
    return pd.DataFrame(
        {'E1': ['E1', 'E2', 'E3', 'E1'],
        'R': ['R1', 'R2', 'R2', 'R1'],
        'E2': ['E2', 'E1', 'E1', 'E3']}
        )

@pytest.fixture
def yes_drop_output():
    return pd.DataFrame(
        {'E1': ['E1', 'E3', 'E1'],
        'R': ['R1', 'R2', 'R1'],
        'E2': ['E2', 'E1', 'E3']}
        )

@pytest.fixture
def no_drop_output():
    return pd.DataFrame(
        {'E1': ['E1', 'E2', 'E3', 'E1'],
        'R': ['R1', 'R2', 'R2', 'R1'],
        'E2': ['E2', 'E1', 'E1', 'E3']}
        )

def test_symmetric_drop_yes(yes_drop_input, yes_drop_output):

    new_df = eu.symmetric_drop(yes_drop_input)

    assert_frame_equal(new_df, yes_drop_output)

def test_symmetric_drop_no(no_drop_input, no_drop_output):

    new_df = eu.symmetric_drop(no_drop_input)

    assert_frame_equal(new_df, no_drop_output)


                        #### check_rels_symmetrically ####

@pytest.fixture
def check_sym_same_labs_preds():
    return pd.DataFrame(
        {'E1': ['hello', 'Sparty', 'I'],
        'R': ['REL', 'REL', 'REL'],
        'E2': ['world', 'I', 'green']}
        )

@pytest.fixture
def check_sym_diff_labs_preds():
    return pd.DataFrame(
        {'E1': ['hello', 'Sparty'],
        'R': ['REL', 'SYMREL'],
        'E2': ['world', 'I']}
        )

@pytest.fixture
def check_sym_gold():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'MSU'],
        'R': ['REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'cool']}
        )

@pytest.fixture
def check_sym_same_labs_check_rels_answer():
    return 2, 1, 1

@pytest.fixture
def check_sym_same_labs_no_check_rels_answer():
    return 2, 1, 1

@pytest.fixture
def check_sym_diff_labs_check_rels_answer():
    return 1, 1, 2

@pytest.fixture
def check_sym_diff_labs_no_check_rels_answer():
    return 2, 0, 1

def test_check_rels_symmetrically_same_labs_check_rels(check_sym_same_labs_preds,
                    check_sym_gold, check_sym_same_labs_check_rels_answer):

    vals = eu.check_rels_symmetrically(check_sym_same_labs_preds,
                        check_sym_gold, True)

    assert vals == check_sym_same_labs_check_rels_answer

def test_check_rels_symmetrically_same_labs_no_check_rels(check_sym_same_labs_preds,
                    check_sym_gold, check_sym_same_labs_no_check_rels_answer):

    vals = eu.check_rels_symmetrically(check_sym_same_labs_preds,
                        check_sym_gold, False)

    assert vals == check_sym_same_labs_no_check_rels_answer

def test_check_rels_symmetrically_diff_labs_check_rels(check_sym_diff_labs_preds,
                    check_sym_gold, check_sym_diff_labs_check_rels_answer):

    vals = eu.check_rels_symmetrically(check_sym_diff_labs_preds,
                    check_sym_gold, True)

    assert vals == check_sym_diff_labs_check_rels_answer

def test_check_rels_symmetrically_diff_labs_no_check_rels(check_sym_diff_labs_preds,
                    check_sym_gold, check_sym_diff_labs_no_check_rels_answer):

    vals = eu.check_rels_symmetrically(check_sym_diff_labs_preds,
                    check_sym_gold, False)

    assert vals == check_sym_diff_labs_no_check_rels_answer


                        #### check_rels_asymmetrically ####

@pytest.fixture
def check_asym_same_labs_preds():
    return pd.DataFrame(
        {'E1': ['hello', 'Sparty', 'I'],
        'R': ['REL', 'REL', 'REL'],
        'E2': ['world', 'I', 'green']}
        )

@pytest.fixture
def check_asym_diff_labs_preds():
    return pd.DataFrame(
        {'E1': ['hello', 'Sparty'],
        'R': ['REL', 'SYMREL'],
        'E2': ['world', 'I']}
        )

@pytest.fixture
def check_asym_gold():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'MSU'],
        'R': ['REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'cool']}
        )

@pytest.fixture
def check_asym_same_rels_answer():
    return 1, 2, 2

@pytest.fixture
def check_asym_diff_rels_answer():
    return 1, 1, 2

def test_check_rels_asymmetrically_same_labs(check_asym_same_labs_preds,
                    check_asym_gold, check_sym_same_rels_answer):

    vals = eu.check_rels_asymmetrically(check_asym_same_labs_preds,
                        check_asym_gold)

    assert vals == check_sym_same_same_rels_answer

def test_check_rels_asymmetrically_same_labs(check_asym_diff_labs_preds,
                    check_asym_gold, check_asym_diff_rels_answer):

    vals = eu.check_rels_asymmetrically(check_asym_diff_labs_preds,
                        check_asym_gold)

    assert vals == check_asym_diff_rels_answer


############################## Integration Tests ##############################

                        ######## dedup_trip_df #########

@pytest.fixture
def dedup_direct_dup_input():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'I', 'hello'],
        'R': ['REL', 'REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'green', 'world']}
        )

@pytest.fixture
def dedup_direct_dup_output():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'I'],
        'R': ['REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'green']}
        )

def test_dedup_trip_df_direct_dup_sym(dedup_direct_dup_input,
                                        dedup_direct_dup_output):

    new_df = eu.dedup_trip_df(dedup_direct_dup_input, True, ['REL'])

    assert_frame_equal(new_df, dedup_direct_dup_output)

def test_dedup_trip_df_direct_dup_asym(dedup_direct_dup_input,
                                        dedup_direct_dup_output):

    new_df = eu.dedup_trip_df(dedup_direct_dup_input, True, [])

    assert_frame_equal(new_df, dedup_direct_dup_output)


@pytest.fixture
def dedup_rev_dup_input():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'I', 'world'],
        'R': ['REL', 'REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'green', 'hello']}
        )

@pytest.fixture
def dedup_rev_dup_sym_output():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'I'],
        'R': ['REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'green']}
        )

@pytest.fixture
def dedup_rev_dup_asym_output():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'I', 'world'],
        'R': ['REL', 'REL', 'REL', 'REL'],
        'E2': ['world', 'Sparty', 'green', 'hello']}
        )

def test_dedup_trip_df_sym(dedup_rev_dup_input, dedup_rev_dup_sym_output):

    new_df = eu.dedup_trip_df(dedup_rev_dup_input, True, ['REL'])

    assert_frame_equal(new_df, dedup_rev_dup_sym_output)

def test_dedup_trip_df_asym(dedup_rev_dup_input, dedup_rev_dup_asym_output):

    new_df = eu.dedup_trip_df(dedup_rev_dup_input, True, [])

    assert_frame_equal(new_df, dedup_rev_dup_asym_output)


@pytest.fixture
def dedup_per_rel_no_drop_input():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'green', 'how', 'much', 'world'],
        'R': ['REL', 'REL', 'SYMREL', 'SYMREL', 'SYMREL', 'REL'],
        'E2': ['world', 'Sparty', 'blue', 'you', 'ado', 'hello']}
        )

@pytest.fixture
def dedup_per_rel_no_drop_output():
    # For the sake of getting the test to pass, this has to be sorted by whether
    # the relation is symmetric or not. Sorting here as opposed to in the code
    # because IRL it doesn't matter what order the resulting triples are in
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'world', 'green', 'how', 'much'],
        'R': ['REL', 'REL', 'REL', 'SYMREL', 'SYMREL', 'SYMREL'],
        'E2': ['world', 'Sparty', 'hello', 'blue', 'you', 'ado']}
        )

def test_dedup_trip_df_per_rel_no_drop(dedup_per_rel_no_drop_input,
                                        dedup_per_rel_no_drop_output):

    new_df = eu.dedup_trip_df(dedup_per_rel_no_drop_input, True, ['SYMREL'])

    assert_frame_equal(new_df, dedup_per_rel_no_drop_output)


@pytest.fixture
def dedup_no_rels_input():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'green', 'how', 'much', 'world'],
        'R': ['REL', 'REL', 'SYMREL', 'SYMREL', 'SYMREL', 'REL'],
        'E2': ['world', 'Sparty', 'blue', 'you', 'ado', 'hello']}
        )

@pytest.fixture
def dedup_no_rels_output():
    return pd.DataFrame(
        {'E1': ['hello', 'I', 'green', 'how', 'much'],
        'R': ['REL', 'REL', 'SYMREL', 'SYMREL', 'SYMREL'],
        'E2': ['world', 'Sparty', 'blue', 'you', 'ado']}
        )

def test_dedup_trip_df_no_rels(dedup_no_rels_input,
                                        dedup_no_rels_output):

    new_df = eu.dedup_trip_df(dedup_no_rels_input, False, ['SYMREL'])

    assert_frame_equal(new_df, dedup_no_rels_output)


                        ##### GetF1 basic functionality ######

@pytest.fixture
def basic_preds():
    return [['hello', 'REL', 'world'],
            ['Sparty', 'REL', 'goodbye'],
            ['Sparty', 'REL', 'ice cream'],
            ['cool', 'REL', 'MSU'],
            ['computer', 'REL', 'nice']]

@pytest.fixture
def basic_gold():
    return [['hello', 'REL', 'world'],
            ['goodbye', 'REL', 'Sparty'],
            ['I', 'REL', 'green'],
            ['Sparty', 'REL', 'ice cream'],
            ['MSU', 'REL', 'cool'],
            ['computer', 'REL', 'annoying']]

@pytest.fixture
def basic_sym_answer():
    return 4, 1, 2

@pytest.fixture
def basic_asym_answer():
    return 2, 3, 4

def test_get_f1_input_basic_sym(basic_preds, basic_gold, basic_sym_answer):

    vals = eu.get_f1_input(basic_preds, basic_gold, True, ['REL'])

    assert vals == basic_sym_answer

def test_get_f1_basic_asym(basic_preds, basic_gold, basic_asym_answer):

    vals = eu.get_f1_input(basic_preds, basic_gold, True)

    assert vals == basic_asym_answer

            ##### GetF1 with mismatched sym/asym rel labels #####

@pytest.fixture
def mismatch_rels_preds():
    return [['hello', 'REL', 'world'],
            ['Sparty', 'REL', 'I'],
            ['blue', 'REL', 'green'],
            ['green', 'REL', 'blue']]

@pytest.fixture
def mismatch_rels_gold():
    return [['hello', 'REL', 'world'],
            ['I', 'REL', 'Sparty'],
            ['green', 'SYMREL', 'blue'],
            ['how', 'SYMREL', 'you'],
            ['much', 'SYMREL', 'ado']]

@pytest.fixture
def mismatch_mini_preds():
    return [['E1', 'R2', 'E2'],
            ['E2', 'R1', 'E1'],
            ['E1', 'R1', 'E3']]

@pytest.fixture
def mismatch_mini_gold():
    return [['E1', 'R1', 'E2'],
            ['E1', 'R2', 'E3']]

@pytest.fixture
def mismatch_rels_answer():
    return 1, 3, 4

@pytest.fixture
def mismatch_mini_answer():
    return 1, 2, 1

def test_get_f1_input_mismatch_rels(mismatch_rels_preds, mismatch_rels_gold,
                                        mismatch_rels_answer):

    vals = eu.get_f1_input(mismatch_rels_preds, mismatch_rels_gold, True,
                            sym_rels=['SYMREL'])

    assert vals == mismatch_rels_answer

def test_get_f1_input_mismatch_mini(mismatch_mini_preds, mismatch_mini_gold,
                                        mismatch_mini_answer):

    vals = eu.get_f1_input(mismatch_mini_preds, mismatch_mini_gold, True, ['R1'])

    assert vals == mismatch_mini_answer


             ###### GetF1 for capitalization and whitespace #######

@pytest.fixture
def cap_whitespace_preds():
    return [['helLO', 'REL', 'world'],
            ['i', 'symrel', 'sparTY'],
            ['much\n', 'Rel', 'ADO']]

@pytest.fixture
def cap_whitespace_gold():
    return [['hello', 'REL', 'world\n'],
            ['I', 'REL', 'Sparty'],
            ['ado', 'SYMREL', 'much']]

@pytest.fixture
def cap_whitespace_check_rels_answer():
    return 1, 2, 2

@pytest.fixture
def cap_whitespace_no_check_rels_answer():
    return 3, 0, 0

def test_get_f1_input_cap_whitespace_check_rels(cap_whitespace_preds,
                        cap_whitespace_gold, cap_whitespace_check_rels_answer):

    vals = eu.get_f1_input(cap_whitespace_preds, cap_whitespace_gold, True,
                            sym_rels=['SYMREL'])

    assert vals == cap_whitespace_check_rels_answer

def test_get_f1_input_cap_whitespace_no_check_rels(cap_whitespace_preds,
                        cap_whitespace_gold, cap_whitespace_no_check_rels_answer):

    vals = eu.get_f1_input(cap_whitespace_preds, cap_whitespace_gold, False)

    assert vals == cap_whitespace_no_check_rels_answer

