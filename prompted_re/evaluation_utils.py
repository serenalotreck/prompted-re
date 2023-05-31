"""
Module with functions to evaluate model performance.

Author: Serena G. Lotreck
"""
import numpy as np
import pandas as pd
import scipy.stats as st


def safe_div(num, denom):
    """
    Function from https://www.github.com/dwadden/dygiepp/dygie/training/f1.py
    """
    if denom > 0:
        return num / denom
    else:
        return 0


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
        onlist = ['E1', 'E2']

    # Check in both directions
    dir1_matches = pd.merge(gold_df, pred_df, how='inner', on=onlist)

    # Swap preds df and check again
    pred_rev = pred_df.iloc[:, ::-1].rename(columns={'E1':'E2', 'E2':'E1'})
    dir2_matches = pd.merge(gold_df, pred_rev, how='inner', on=onlist)

    # Get numbers
    tp = len(dir1_matches) + len(dir2_matches)
    fp = len(pred_df) - tp
    fn = len(gold_df) - tp

    return tp, fp, fn

def symmetric_drop(df):
    """
    Drop rows for suplicates including symmetrically reversed rows. Considers
    relation type; this is designed to get rid of true duplicates, not
    artificial duplicates introduced by not considering relation types.

    parameters:
        df, pandas df: df from which to drop duplicates

    returns:
        dropped_df, pandas df: updated df
    """
    # Make a column with the sorted row as a column
    df['sorted_row'] = [sorted([a, b, c]) for a, b, c in zip(
                                                df['E1'], df['R'], df['E2'])]

    # Turn the new column into a string
    df['sorted_row'] = df['sorted_row'].astype(str)

    # Drop duplicates
    dropped_df = df.drop_duplicates(subset=['sorted_row'])

    # Drop the additional row
    dropped_df = dropped_df.drop(columns='sorted_row')

    # Reset index
    dropped_df = dropped_df.reset_index(drop=True)

    return dropped_df


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
        deduped_df, pandas df: deduplicated df
    """
    # Get reltypes and check if they're all symmetrical
    rel_types = df['R'].unique()
    in_sym = [r in sym_rels for r in rel_types]
    # If we don't care about order or all rels are symmetric
    if not check_rels or all(in_sym):
        deduped_df = symmetric_drop(df)
    # If there are any rels that are not symmetric, go through by rel type
    else:
        deduped_df_list = []
        for rel_type in rel_types:
            df_sub = df[df['R'] == rel_type].copy(deep=True)
            if rel_type in sym_rels:
                deduped_df_sub = symmetric_drop(df_sub)
                deduped_df_list.append(deduped_df_sub)
            else:
                deduped_df_sub = df_sub.drop_duplicates()
                deduped_df_list.append(deduped_df_sub)
        deduped_df = pd.concat(deduped_df_list)
        deduped_df = deduped_df.reset_index(drop=True)
        

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
    sym_rels = [s.lower() for s in sym_rels] if sym_rels is not None else []

    # Check if either the gold or the predictions are empty
    if len(preds) == 0 or len(gold) == 0:
        tp = 0
        fp = len(preds)
        fn = len(gold)

        return tp, fp, fn

    else:
        # Make triple dataframes
        pred_df = pd.DataFrame(np.array(preds), columns=['E1', 'R', 'E2'])
        gold_df = pd.DataFrame(np.array(gold), columns=['E1', 'R', 'E2'])
    
        # Lowercase everything and strip leading/training whitespace
        for col_name in ['E1', 'R', 'E2']:
            pred_df[col_name] = pred_df[col_name].str.lower().str.strip()
            gold_df[col_name] = gold_df[col_name].str.lower().str.strip()
    
        # Deduplicate
        orig_lens = [len(pred_df), len(gold_df)]
        pred_df = dedup_trip_df(pred_df, check_rels, sym_rels)
        gold_df = dedup_trip_df(gold_df, check_rels, sym_rels)
        # print(f'After deduplication, there are {orig_lens[0] - len(pred_df)} '
        #       f'fewer predicted triples, and {orig_lens[1] - len(gold_df)} '
        #       'fewer triples in the gold standard.')
    
        # Check whether we have to go yb rel label or if we can do them all together
        tp, fp, fn = [0, 0, 0]
        rel_types = gold_df['R'].unique()
        in_sym = [r in sym_rels for r in rel_types]
        # If we don't care about relation label, can do the entire thing at once:
        if not check_rels:
            new_tp, new_fp, new_fn = check_rels_symmetrically(pred_df, gold_df, check_rels)
            tp += new_tp
            fp += new_fp
            fn += new_fn
        # We also can do everything at once if all relation types are symmetrical
        elif all(in_sym):
            new_tp, new_fp, new_fn = check_rels_symmetrically(pred_df, gold_df, check_rels)
            tp += new_tp
            fp += new_fp
            fn += new_fn
        # Otherwise, we have to subset by type and perform operations on each subset
        else:
            for rel_type in rel_types:
                # Subset
                pred_sub = pred_df.loc[pred_df['R'] == rel_type]
                gold_sub = gold_df.loc[gold_df['R'] == rel_type]
                # Check if the rel type we're on is symmetrical
                if rel_type in sym_rels:
                    new_tp, new_fp, new_fn = check_rels_symmetrically(pred_sub, gold_sub, check_rels)
                    tp += new_tp
                    fp += new_fp
                    fn += new_fn
                else:
                    new_tp, new_fp, new_fn = check_rels_asymmetrically(pred_sub, gold_sub)
                    tp += new_tp
                    fp += new_fp
                    fn += new_fn

        return tp, fp, fn


def calculate_f1(preds, gold, check_rels=True, sym_rels=None):
    """
    Calculate F1 score.

    parameters:
        preds, list of list of str: predicted triples
        gold, list of list of str: gold triples
        check_rels, bool: whether or not to consider the identity of relation
            labels, default is True
        sym_rels, list of str: list of relations that should be evaluated
            order-agnostically, default is None

    returns:
        f1, float: F1 score for the provided sample
    """
    # Get tp/fp/fn counts
    tp, fp, fn = get_f1_input(preds, gold, check_rels=True, sym_rels=None)

    # Add to get f1 inputs
    predicted = tp + fp
    gold = tp + fn
    matched = tp

    # Calculate F1
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return f1
    

def calculate_performance(dset_df, bootstrap=True, bootstrap_iters=500,
                            check_rels=True, sym_rels=None):
    """
    Compute the F1 score of a model on a given dataset. For test sets with fewer
    than 30 documents, uses the t distribution, and for larger than 30 documents,
    uses the normal distribution.

    parameters:
        dset_df, pandas df: rows are documents, must have the columns 'trips'
            and 'preds', where trips are gold standard and preds are predictions,
            and they are both formatted as lists of list
        bootstrap, bool: whether or not to bootstrap performance estimate,
            default is True
        bootstrap_iters, int: how many bootstrap samples to pull, default is 500
        check_rels, bool: whether or not to consider relation label identity,
            default is True
        sym_rels, list of str: relations to treat as symmetric, default is None

    return:
        f1_mean, float: F1 score
        CI, tuple of float or None: CI if bootstrap=True, else None
    """
    # Bootstrap
    if bootstrap:
        sample_f1s = []
        for i in range(bootstrap_iters):
            # Draw sample
            sample = dset_df.sample(n=len(dset_df), replace=True)

            # Compute statistic on sample
            # This gets a separate F1 for each document
            sample_f1 = sample.apply(lambda row: calculate_f1(row['preds'],
                                            row['trips'], check_rels=check_rels,
                                            sym_rels=sym_rels), axis=1)

            # Average the F1 for this sample
            sample_f1 = np.mean(sample_f1)

            # Add to the sample list
            sample_f1s.append(sample_f1)
        print(sample_f1s)

        # Generate the CI for this metric
        if len(sample_f1s) <= 30:
            CI = st.t.interval(alpha=0.95, df=len(sample_f1s)-1,
              loc=np.mean(sample_f1s),
              scale=st.sem(sample_f1s))
        else:
            CI = st.norm.interval(alpha=0.95,
                 loc=np.mean(sample_f1s),
                 scale=st.sem(sample_f1s))
        # Get the mean value of the metric
        f1_mean = np.mean(sample_f1s)
    # Get point estimate if no bootstrapping
    else:
        # get the F1s for all the docs
        doc_f1s = dset_df.apply(lambda row: calculate_f1(row['preds'],
                                            row['trips'], check_rels=check_rels,
                                            sym_rels=sym_rels), axis=1)
        # Average to get point estimate
        f1_mean = np.mean(doc_f1s)
        CI = None

    return f1_mean, CI
            











