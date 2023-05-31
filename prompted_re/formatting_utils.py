"""
Formatting utilities.

Author: Serena G. Lotreck
"""
import pandas as pd
import jsonlines


def format_dygiepp_df(dygiepp_path, filter_type):
    """
    Get the gold and predicted triples from a dygiepp-formatted dataset. Assumes
    that the keys "relations" and "predicted_relations" are both present.

    parameters:
        dygiepp_path, str: path to dygiepp file to parse
        filter_type, list of str: relation types to keep

    returns:
        dygiepp_df, pandas df: columns are "doc_key", "trips", and "preds
    """
    # Read in the file
    with jsonlines.open(dygiepp_path) as reader:
        dygiepp_dset = []
        for obj in reader:
            dygiepp_dset.append(obj)

    # Go through and format
    dropped = 0
    dset_dict = {'doc_key': [], 'trips': [], 'preds': []}
    for doc in dygiepp_dset:
        doc_key = doc["doc_key"]
        all_toks = [tok for sent in doc["sentences"] for tok in sent]
        gold_doc_trips = []
        for i, sent in enumerate(doc["sentences"]):
            for rel in doc["relations"][i]:
                ent1 = all_toks[rel[0]:rel[1]+1]
                if isinstance(ent1, list):
                    ent1 = ' '.join(ent1)
                ent2 = all_toks[rel[2]:rel[3]+1]
                if isinstance(ent2, list):
                    ent2 = ' '.join(ent2)
                rel_type = rel[4]
                if rel_type not in filter_type:
                    dropped += 1
                    continue
                rel_list = [ent1, rel_type, ent2]
                gold_doc_trips.append(rel_list)
            doc_preds = []
            for rel in doc["predicted_relations"][i]:
                ent1 = all_toks[rel[0]:rel[1]+1]
                if isinstance(ent1, list):
                    ent1 = ' '.join(ent1)
                ent2 = all_toks[rel[2]:rel[3]+1]
                if isinstance(ent2, list):
                    ent2 = ' '.join(ent2)
                rel_type = rel[4]
                if rel_type not in filter_type:
                    dropped += 1
                    continue
                rel_list = [ent1, rel_type, ent2]
                doc_preds.append(rel_list)
        dset_dict['doc_key'].append(doc_key)
        dset_dict['trips'].append(gold_doc_trips)
        dset_dict['preds'].append(doc_preds)

    # Make df
    dygiepp_df = pd.DataFrame(dset_dict)

    return dygiepp_df, dropped
        
