"""
Run a prompt against a dataset for a given class of models.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
from datasets import load_dataset
from transformers import pipeline
import yaml


def format_prompt_make_pred(pipe, dset_split, yaml_data,
                            fewshot_example_dict=None):
    """
    For each document in a given split, format the prompt and pass to the LLM
    to get results.
    
    parameters:
        pipe, HuggingFace TextGenerationPipeline: pipeline with model
        dset_split, Huggingface Dataset: splt of the dataset to use
        yaml_data, dict: dictionary of special token and context prompt to use
        fewshot_example_dict, dict: keys are 'input' and 'output', values are
            lists containing input and output pairs at corresponding indices,
            None if no file was passed

    returns:
        preds, <dtype>: Model predictions
    """
    # Format fewshot examples if there are any
    if fewshot_example_dict is not None:
        examples = yaml_data['context']
        for inp, out in zip(fewshot_example_dict['input'], fewshot_example_dict['output']):
            examp_str = yaml_data['turn_template']
            examp_str.replace('user', yaml_data['user'])
            examp_str.replace('user-message', inp)
            examp_str.replace('bot', yaml_data['bot'])
            examp_str.replace('bot-message', out)
            examples += examp_str

    # For each document
    for doc in dset_split:
        # Add 
        

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
        trip = [arg1_text, label, arg2_text]
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


def main(model, dataset, special_yaml, out_loc, out_prefix, fewshot_examples,
            verbose):

    # Load the dataset and the model
    verboseprint('\nLoading in dataset...')
    full_dset = load_dataset(dataset)
    verboseprint(f'Dataset {dataset} contains the following splits: '
                f'{full_dset.keys()}')
    vseboseprint('\nLoading in the model as a text generation pipeline...')
    pipe = pipeline('text-generation', model=model)

    # Load the yaml and the fewshot examples
    verboseprint('\nLoading in the special token and context prompt yaml...')
    with open(special_yaml, 'r') as stream:
        yaml_data = pyyaml.safe_load(stream)
    if fewshot_examples is not None:
        verboseprint('\nLoading fewshot examples...')
        with open(fewshot_examples) as myf:
            fewshot_example_dict = json.load(myf)

    # Format dataset to inputs & outputs
    verboseprint('\nFormatting dataset to inputs and outputs...')
    process_dset(full_dset, dataset)

    # Format prompts and make predictions with LLM
    verboseprint('\nFormatting prompts and getting predictions...')
    for split in full_dset.keys():
        if split != 'sample':
            verbosprint(f'On split {split}...')
            preds = format_prompt_make_pred(pipe, full_dset[split], yaml_data,
                                            fewshot_example_dict)

    # Evaluate predictions

    # Save predictions and results
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run prompts')

    parser.add_argument('model', type=str,
            help='huggingface model string to test')
    parser.add_argument('dataset', type=str,
            help='huggingface dataset string to test against')
    parser.add_argument('special_yaml', type=str,
            help='Path to yaml file that specifies special tokens and context '
            'prompt for the given model')
    parser.add_argument('out_loc', type=str,
            help='Path to save outputs')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to output files')
    parser.add_argument('-fewshot_examples', type=str,
            help='Path to a file containing fewshot input/output examples')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    args.special_yaml = abspath(args.special_yaml)
    args.out_loc = abspath(args.out_loc)
    if args.fewshot_examples is not None:
        args.fewshot_examples = abspath(args.fewshot_examples)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(**vars(args))    