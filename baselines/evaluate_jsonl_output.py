"""
Script to compute the F1 for model predictions in dygiepp format.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import json
import sys
sys.path.append('../prompted_re')
from formatting_utils import format_dygiepp_df
from evaluation_utils import calculate_performance


def main(dygiepp_path, eval_config):

    # Read in eval_config
    verboseprint('\nLoading in the evaluation config...')
    with open(evaluation_config) as myf:
        eval_config = json.load(myf)
    key_check = [k in eval_config.keys() for k in ['bootstrap',
                                                'bootstrap_iters', 'check_rels',
                                                'sym_rels', 'filter_type']]
    assert all(key_check), ('One or more required keys is missing from the '
                            'evaluation configuration, please try again.')

    # Format the dygiepp dataset
    verboseprint('\nReading in predictions and gold standard...')
    dset_df, dropped = format_dygiepp_df(dygiepp_path, eval_config['filter_type'])
    print('{dropped} relations were dropped because their types were not '
            'included in "filter_type"')

    # Calculate performance
    verboseprint('\nCalculating performance...')
    f1, CI = calculate_performance(dset_df,
                                    bootstrap=eval_config['bootstrap'],
                                    bootstrap_iters=eval_config['bootstrap_iters'],
                                    check_rels=eval_config['check_rels'],
                                    sym_rels=eval_config['sym_rels']
                                    )

    # Print result
    print('F1: {f1}, CI: {CI}\n\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate jsonl output')

    parser.add_argument('dygiepp_path', type=str,
            help='Path to file to evaluate. Must contain both relations and '
            'predicted_relations fields')
    parser.add_argument('evaluation_config', type=str,
            help='Path to evaluation config file')
    parser.add-argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.arse_args()

    args.dygiepp_path = abspath(args.dygiepp_path)
    args.evaluation_config = abspath(args.evaluation_config)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(args.dygiepp_path, args.evaluation_config)