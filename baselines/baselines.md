# Relation Extraction Baselines

## End-to-End RE
The first baseline we implement is the [end-to-end-ChemProt]() approach.  In order to fairly compare all baselines, we use the same processed version of the ChemProt dataset for training all baselines,, which is presented in the [manuscript for this approach](https://arxiv.org/pdf/2304.01344.pdf). The processing pipeline for this version of the dataset uses a tokenization approach that minimizes the loss of annotations when converting the dataset to its tokenized form. The need for tokenization-linked annotations is one of the major limitations for these baseline methods, since the conversion from formats like the commonly-used [brat standoff](https://brat.nlplab.org/standoff.html) format often results in tokenization mismatches that can cause data loss.

TODO how we did this

## DyGIE++
The second baseline we implement is the [DyGIE++ algorithm](https://www.semanticscholar.org/reader/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8). In order to run the DyGIE++ baseline, we performed the following steps:

1. Clone the [dygiepp](https://github.com/dwadden/dygiepp/) repository and `cd` into it
2. Follow the instructions in the [dependencies](https://github.com/dwadden/dygiepp/tree/master#dependencies) section of the docs
3. Make the `jsonnet` configuration file in the `dygiepp/training_config` directory. The actual contents of the configuration file required to train the model will depend on the location of your local copy of the end-to-end repository. The configuration file we used, with indications of where to put your local paths, is:

```
local template = import "template.libsonnet";
 
template.DyGIE {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 2,
  data_paths: {
    train: "your/path/to/end-to-end-ChemProt/chemprot_data/processed_data/json/train.json",
    validation: "your/path/to/end-to-end-ChemProt/chemprot_data/processed_data/json/dev.json",
    test: "your/path/to/end-to-end-ChemProt/chemprot_data/processed_data/json/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 25
  },
}
``` 
This is identical to the original `chemprot.jsonnet` included in the `dygiepp` repo, just with the paths to the dataset splits replaced.

4. Assuming your training configuration file is named `chemprot_e2e.jsonnet` and is located in the `dygiepp/training_config` directory, from the root `dygiepp` directory, run the following:

```
bash scripts/train.sh chemprot_e2e.jsonnet
```

5. Once the model is trained, run the following code to apply the model to the test set:
