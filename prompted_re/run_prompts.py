"""
Run a prompt against a dataset for a given class of models.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from evaLuation_utils import calculate_performance


def save_preds(full_dset, out_loc, out_prefix):
    """
    Save the predictions as a json.

    parameters:
        full_dset, DatasetDict: dataset with preds
        out_loc, str: lcoation to save
        out_prefix, str: string to prepend to outputs

    returns:
        save_names, list of str: full paths to saved files
    """
    save_names = []
    for split, dset in full_dset.items():
        dset_preds = dset['preds']
        dset_pmids = dset['pmid'] ## TODO generalize
        out_dict = {k:v for k,v in zip(dset_pmids, dset_preds)}
        out_name = f'{out_loc}/{out_prefix}_{split}_{predictions}.json'
        save_names.append(out_name)
        with open(out_name) as myf:
            json.dump(out_dict, myf)

    return save_names


def evaluate_preds(full_dset, out_loc, out_prefix):
    """
    Evaluate and save model predictions.

    parameters:
        full_dset, DatasetDict: dataset with preds
        out_loc, str: lcoation to save
        out_prefix, str: string to prepend to outputs

    returns:
        eval_name, str: path to saved file
    """
    eval_dict = {}
    for split, dset in full_dset.items():
        f1, CI


def format_preds(response):
    """
    Extract triples from a given pipe response.

    parameters:
        response, list of dict: responses from text-generation pipeline object
    """
    print(response)


def format_prompt_make_pred(model, tokenizer, dset_split, yaml_data, prompt,
                            fewshot_example_dict=None):
    """
    For each document in a given split, format the prompt and pass to the LLM
    to get results.

    parameters:
        model, huggingface AutoModelForCausalLM: model to use for inference
        tokenizer, huggingface AutoTokenizer: tokenizer to use
        dset_split, Huggingface Dataset: splt of the dataset to use
        yaml_data, dict: dictionary of special token and context prompt to use
        prompt, str: Prompt to use. Should be formatted to that the target text
            can be directly appended to the string.
        fewshot_example_dict, dict: keys are 'input' and 'output', values are
            lists containing input and output pairs at corresponding indices,
            None if no file was passed

    returns:
        None, modifies dataset in place to add predictions
    """
    # Format fewshot examples if there are any
    if fewshot_example_dict is not None:
        examples = yaml_data['context']
        for inp, out in zip(fewshot_example_dict['input'], fewshot_example_dict['output']):
            examp_str = yaml_data['turn_template']
            examp_str.replace('<|user|>', yaml_data['user'])
            examp_str.replace('<|user-message|>', inp)
            examp_str.replace('<|bot|>', yaml_data['bot'])
            examp_str.replace('<|bot-message|>', out)
            examples += examp_str

    # For each document
    preds = []
    for doc in dset_split:
        # Add one more user prompt and the doc text
        final_str = yaml_data['turn_template']
        bot_msg_idx = final_str.index('<|bot-message|>')
        final_str = final_str[:bot_msg_idx]
        final_str.replace('<|user|>', yaml_data['user'])
        final_str.replace('<|user-message|>', prompt + doc['text'])
        final_str.replace('<|bot|>', yaml_data['bot'])
        if fewshot_example_dict is not None:
            final_str = examples + final_str

        # Generate predictions
        inputs = tokenizer(final_str, return_tensors='pt')
        inputs = inputs.to(0)
        output = model.generate(inputs['input_ids'])
        response = tokenizer.decode(output[0].tolist())

        # Format the output
        doc_preds = format_preds(response)

        # Append to predictions
        preds.append(doc_preds)

    # Add to dataset
    dset_split.add_column('preds', preds)


def get_chemprot_trips(doc):
    """
    Gets the triples for a given ChemProt dataset document.

    parameters:
        doc, dict: one document from the ChemProt dataset

    returns:
        trips, list of list: triples
    """
    trips = []
    for i in range(len(doc['relations']['type'])):
        label = doc['relations']['type'][i]
        arg1 = doc['relations']['arg1'][i]
        arg1_idx = doc['entities']['id'].index(arg1)
        arg1_txt = doc['entities']['text'][arg1_idx]
        arg2 = doc['relations']['arg1'][i]
        arg2_idx = doc['entities']['id'].index(arg2)
        arg2_txt = doc['entities']['text'][arg2_idx]
        trip = [arg1_txt, label, arg2_txt]
        trips.append(trip)

    return trips


def process_dset(full_dset, dataset):
    """
    Processes a relation dataset to have a column containing the unbound correct
    triples for evaluation. This function is likely where dataset idiosyncracies
    will be most relevant, so am preemtively accepting a dataset parameter.

    parameters:
        full_dset, HuggingFace DatasetDict: full dataset to process
        dataset, str: Name of the dataset being processed

    returns:
        None, modifies in place
    """
    # For each split
    for split in full_dset.keys():
        if split != 'sample':
            if dataset == 'bigbio/chemprot':
                trip_col = []
                for doc in full_dset[split]:
                    trip_col.append(get_chemprot_trips(doc))
                full_dset[split].add_column('trips', trip_col)


def main(model, dataset, prompt_path, special_yaml, evaluation_config, out_loc,
        out_prefix, fewshot_examples, verbose):

    # Load the dataset
    verboseprint('\nLoading in dataset...')
    full_dset = load_dataset(dataset)
    verboseprint(f'Dataset {dataset} contains the following splits: '
                f'{full_dset.keys()}')

    # Load in the prompt
    verboseprint('\nLoading in the prompt file...')
    with open(prompt_path) as myf:
        prompt = myf.read()
    print_len = min(len(prompt), 50)
    verboseprint(f'Snapshot of the loaded prompt: {prompt[:print_len]}')

    # Load the yaml and the fewshot examples
    verboseprint('\nLoading in the special token and context prompt yaml...')
    with open(special_yaml, 'r') as stream:
        yaml_data = yaml.safe_load(stream)
    if fewshot_examples is not None:
        verboseprint('\nLoading fewshot examples...')
        with open(fewshot_examples) as myf:
            fewshot_example_dict = json.load(myf)
    else: fewshot_example_dict = None

    # Load evaluation config
    verboseprint('\nLoading in the evaluation config...')
    with open(evaluation_config) as myf:
        eval_config = json.load(myf)
    key_check = [k in eval_config.keys() for k in ['bootstrap',
                                                'bootstrap_iters', 'check_rels',
                                                'sym_rels']]
    assert all(key_check), ('One or more required keys is missing from the '
                            'evaluation configuration, please try again.')

    # Load in the model
    verboseprint('\nLoading in the model and tokenizer...')
    checkpoint = model
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
        torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Format dataset to inputs & outputs
    verboseprint('\nFormatting dataset to inputs and outputs...')
    process_dset(full_dset, dataset)

    # Format prompts and make predictions with LLM
    verboseprint('\nFormatting prompts and getting predictions...')
    for split in full_dset.keys():
        verboseprint(f'On split {split}...')
        format_prompt_make_pred(model, tokenizer, full_dset[split], yaml_data, prompt,
                                            fewshot_example_dict)

    # Evaluate predictions
    verboseprint('\nEvaluating predictions....')
    eval_name = evaluate_preds(full_dset, out_loc, out_prefix)
    verboseprint(f'Evaluations have been saved to {eval_name}')

    # Save predictions
    verboseprint('\nSaving predictions...')
    save_names = save_preds(full_dset, out_loc, out_prefix)
    save_names = [basename(n) for n in save_names]
    verboseprint(f'Evaluations have been saved to the directory {out_loc}, '
                f'with the filenames {", ".join(save_names)}')

    verboseprint('\nDone!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run prompts')

    parser.add_argument('model', type=str,
            help='huggingface model string to test')
    parser.add_argument('dataset', type=str,
            help='huggingface dataset string to test against')
    parser.add_argument('prompt_path', type=str,
            help='Path to a txt file containing the prompt string')
    parser.add_argument('special_yaml', type=str,
            help='Path to yaml file that specifies special tokens and context '
            'prompt for the given model')
    parser.add_argument('evaluation_config', type=str,
            help='Path to a json file specifying the parameters for evaluation')
    parser.add_argument('out_loc', type=str,
            help='Path to save outputs')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to output files')
    parser.add_argument('-fewshot_examples', type=str,
            help='Path to a file containing fewshot input/output examples')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    args.prompt_path = abspath(args.prompt_path)
    args.special_yaml = abspath(args.special_yaml)
    args.out_loc = abspath(args.out_loc)
    args.evaluation_config = abspath(args.evaluation_config)
    if args.fewshot_examples is not None:
        args.fewshot_examples = abspath(args.fewshot_examples)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(**vars(args))
