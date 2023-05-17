# prompted-re
---------------------
Codebase for applying relation extraction prompts to various Open LLM models and datasets from the huggingface `transformers` and `datasets` libraries.

n-ary relation extraction (RE) is defined as identifying *(subject, predicate, object)* triples from natural language text for some set of predicate labels. The RE task has many downstream applications, such as knowledge graph construction, and cannot be trivially solved by LLMs like GPT-4. This repository explores the application of various open-source LLMs to the RE task.

## Models
This codebase can be used with any LLM from the HuggingFace `Transformers` library that is configured for a [`text-generation` pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline).

### Special token and context prompt specifications
Every Open LLM requires a specific set of special tokens in its prompts, as well as a context/system prompt, in order to get reasonable results. These special tokens and context prompts are not necessarily evident in the documentation for these models, which can make application to a new task via prompting difficult.

Some kind folks at HuggingFace pointed me in the direction of a [repository of `yaml` configuration files](https://github.com/oobabooga/text-generation-webui/tree/main/characters/instruction-following) containing special tokens and prompts for a set of Open LLMs. In this repository, yaml files will be used to specify the correct tokens and prompts for a given model. We provide some of these in the `model_prompt_configs` directory.

## Datasets
The intention of this codebase is to allow for evaluation of any suitable model against any Huggingface `datasets` dataset containing n-ary relation annotations. As a starting point, we will use the [ChemProt](https://huggingface.co/datasets/bigbio/chemprot/) dataset, so the code here may be idiosyncratically specific to the ChemProt dataset.

## Prompts
Eventually, we would like to be able to evaluate prompts of an arbitrary number of steps. For the moment, however, we are only able to accept prompts that make a single request, with an arbitrary number of fewshot input/output examples. Examples of the kinds of prompts currently accepted can be found in `re_prompts`.

## Fewshot Examples
s
