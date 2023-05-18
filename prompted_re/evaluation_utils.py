"""
Module with functions to evaluate model performance.

Author: Serena G. Lotreck
"""
import numpy as np
import pandas as pd


def check_rels_asymmetrically(pred_df, gold_df):
    """
    Get TP, FP, and FN asymmetrically.

    parameters:
        pred_df, pandas df: predicted triples
        gold_df, pandas df: gold triples

    returns:
        tp, int: number of true positives
        fp, int: number of false positives
        fn, int: number of false negatives
    """
    # Get matches
    matches = pd.merge(gold_df, pred_df, how='inner', on=['E1', 'R', 'E2'])

    # Get numbers
    tp = len(matches)
    fp = len(pred_df) - tp
    fn = len(gold_df) - tp

    return tp, fp, fn


def check_rels_symmetrically(pred_df, gold_df, check_rels):
    """
    Get TP, FP, and FN symmetrically.

    parameters:
        pred_df, pandas df: predicted triples
        gold_df, pandas df: gold triples
        check_rels, bool: whether or not to consider the identity of relation
            labels

    returns:
        tp, int: number of true positives
        fp, int: number of false positives
        fn, int: number of false negatives
    """
    # Decide which columns we care about
    if check_rels:
        onlist = ['E1', 'R', 'E2']
    else:
        onlist = ['E1', 'E2'

    # Check in both directions
    dir1_matches = pd.merge(gold_df, pred_df, how='inner', on=['E1', 'R', 'E2'])

    # Swap preds df and check again
    pred_rev = pred.iloc[:, ::-1].rename(columns={'E1':'E2', 'E2':'E1'})
    dir2_matches = pd.merge(gold_df, pred_rev, how='inner', on=['E1', 'R', 'E2'])

    # Get numbers
    tp = len(dir1_matches) + len(dir2_matches)
    fp = len(pred_df) - tp
    fn = len(gold_df) - tp

    return tp, fp, fn


def dedup_trip_df(df, check_rels, sym_rels):
    """
    Eliminate identidal triples, considering symmetry if relevant.

    parameters:
        df, pandas df: columns are E1, R and E2
        check_rels, bool: whether or not to consider the identity of relation
            labels, default is True
        sym_rels, list of str: list of relations that should be evaluated
            order-agnostically, default is None

    returns:
        deduped_df, padnas df: deduplicated df
    """
    # Get reltypes and check if they're all symmetrical
    rel_types = gold_df['R'].unique()
    in_sym = [r in sym_rels for r in rel_types]
    # If we don't care about order or all rels are symmetric
    if not check_rels or in_sym.all():
        deduped_df = pd.DataFrame(
            np.sort(df.values, axis=1),
            index=df.index,
            columns=df.columns).drop_duplicates()
    # If there are any rels that are not symmetric, go through by rel type
    else:
        deduped_df_list = []
        for rel_type in rel_types:
            df_sub = df[df['R'] == rel_type]
            if rel_type in sym_rels:
                deduped_df_sub = pd.DataFrame(
                    np.sort(df.values, axis=1),
                    index=df.index,
                    columns=df.columns).drop_duplicates()
                deduped_df_list.append(deduped_df_sub)
            else:
                deduped_df_sub = df.drop_duplicates()
                deduped_df_list.append(deduped_df_sub)
        deduped_df = pd.concat(deduped_df_list)

    return deduped_df


def get_f1_input(preds, gold, check_rels=True, sym_rels=None):
    """
    Given a list of predictions and a list of gold standards, calculates true
    and false positives and false negatives.

    parameters:
        preds, list of list of str: predicted triples
        gold, list of list of str: gold triples
        check_rels, bool: whether or not to consider the identity of relation
            labels, default is True
        sym_rels, list of str: list of relations that should be evaluated
            order-agnostically, default is None

    returns:
        tp, int: number of true positives
        fp, int: number of false positives
        fn, int: number of false negatives
    """
    # Get sym_rels as a list if it was None
    sym_rels = sym_rels if sym_rels is not None else []

    # Make triple dataframes
    pred_df = pd.DataFrame(np.array(preds), columns=['E1', 'R', 'E2'])
    gold_df = pd.DataFrame(np.array(gold), columns=['E1', 'R', 'E2'])

    # Lowercase everything and strip leading/training whitespace
    for col_name in ['E1', 'R', 'E2']:
        pred_df[col_name] = pred_df[col_name].str.lower().str.strip()
        gold_df[col_name] = pred_df[col_name].str.lower().str.strip()

    # Deduplicate
    orig_lens = [len(pred_df), len(gold_df)]
    pred_df = dedup_trip_df(pred_df, check_rels, sym_labs)
    gold_df = dedup_trip_df(gold_df, check_rels, sym_labs)
    print(f'After deduplication, there are {len(orig_lens[0]) - len(pred_df)} '
          f'fewer predicted triples, and {len(orig_lens[1]) - len(gold_df)} '
          'fewer triples in the gold standard.')

    # Check whether we have to go yb rel label or if we can do them all together
    tp, fp, fn = 0
    rel_types = gold_df['R'].unique()
    in_sym = [r in sym_rels for r in rel_types]
    # If we don't care about relation label, can do the entire thing at once:
    if not check_rels:
        tp, fp, fn += check_rels_symmetrically(pred_sub, gold_sub, check_rels)
    # We also can do everything at once if all relation types are symmetrical
    elif in_sym.all():
        tp, fp, fn += check_rels_symmetrically(pred_sub, gold_sub, check_rels)
    # Otherwise, we have to subset by type and perform operations on each subset
    else:
        for rel_type in rel_types:
            # Subset
            pred_sub = pred_df.loc[pred_df['R'] == rel_type]
            gold_sub = gold_df.loc[gold_df['R'] == rel_type]
            # Check if the rel type we're on is symmetrical
            if rel_type in sym_rels:
                tp, fp, fn += check_rels_symmetrically(pred_sub, gold_sub, check_rels)
            else:
                tp, fp, fn += check_rels_asymmetrically(pred_sub, gold_sub)

    return tp, fp, fn














