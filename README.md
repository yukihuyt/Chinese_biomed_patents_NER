# Chinese Biomedical Patents NER
This repository includes the built datasets, source codes and trained models of our study ...
File Structure

## Configuration
Important requirements:


Enviornment files in `\env`.
Configure the python environment using:
```bash
#using pip
pip install --upgrade pip
pip install -r .\env\cbp.txt

#using conda
conda env create -f .\env\cbp.yml
```

## Data Information
BC and HG are two unlabled Chinese Biomedical Patents datasets, which were both retrieved from Google Patents Databases. BC ... HG ...

Part BC and HG were part of both datasets, which were randomly selected from BC and HG and finally used in our Language Model fine-tuning experiments (Mixed_LM) for efficiency considerations.

Demo is a small demo set to quickly check your configuration and do some sanity check before your real experiments.

Cbp_gold is our finally built gold standard dataset. It contains...
It was labeled with BIO format labels, each line contains one character and its corresponding BIO tag.
`cbp_gold_total.bio` is the original version of the gold standard set, while `no_long_cbp_gold_total.bio` makes sure each setence is not longer than 500 characters (excceded sentences will be split into sub-sentences.)
The included 5 folders contain the evaluation sets we built from the gold standard set, which were finally applied in all our NER evaluation experiments.

## Run experiments
The folders `\Supervised_Original`, `\BERT_LM_mixed` and `\partBERT_CRF` include the source codes of corresponding training methods.

To make sure your configurations are sufficient and ready, you can run a demo experiment by using:
```bash
run demo.sh
```

For each training method, to reproduce our final experiments, directly run the `.sh` file. You need to change the parameter of `.sh` to run your costomized expriments. Some important parameters are... 

