# Chinese Biomedical Patents NER

This repository includes the built datasets, source codes and trained models of our study **Named Entity Recognition for Chinese biomedical patents**.

## Configuration
Enviornment files in `/env`.  
Configure the python environment using:
```bash
#using pip
pip install --upgrade pip
pip install -r ./env/cbp.txt

#using conda
conda env create -f ./env/cbp.yml
```

## Datasets
The `/data` contains all our built datasets.  
Detailed information see [dataset information](./data/README.md)

## Models
Here we release 3 models generated during our experiments which you can use to either reproduce our results or run your own customized experiments.

`/partHG_1epoch` is the bert-base-chinese model been fine-tuned on our large unlabeled dataset `HG` for 1 epoch, while `/partBC_30epochs` on unlabeled dataset `BC` for 30 epochs. These 2 fine-tuned lanaguage model can be used to run your customized NER experiments or be loaded to continue fine-tuning on more data/epochs.

`/final_trained` is the NER model trained on our whole labeled dataset `cbp_gold`, using the fine-tuned language model `/partHG_1epoch`. It is also our final selected model to generate predictions on unlabeled dataset for further analysis. This NER model is possible to be directly applied on new data to generate NER predictions.

## Run experiments
The directories `/Supervised_Original`, `/BERT_LM_mixed` and `/partBERT_CRF` contain the source codes of corresponding training methods described in our paper.

To make sure your configurations are sufficient and ready, you can run a demo NER experiment for only 1 epoch by simply running `demo_run.sh` of each method.

**Reproduce our results**  
To reproduce our final evaluation experiments of `/Supervised_Original` and `/partBERT_CRF`, change the GPU resource related parameters of each `iter_exp.sh` file, like available GPU id and possible batch size, then simply run it. 


For `/BERT_LM_mixed` method, first you need to download the bin files of each model in correspoding directories.

* Download [partHG_1epoch_bin](https://ufile.io/ijfrwomm) under `./models/partHG_1epoch`
* Download [...]() under `./models/partBC_30epochs`
* Download [...]() under `./models/final_trained`  
(Links not ready)

There are 2 sub-directories under `/BERT_LM_mixed`.  
The `/lmft_example` contains codes to fine-tune the `bert-base-chinses`, while `/ner_example` contains codes to run NER experiments using fine-tuned language models.

To reproduce the evaluation results of this method, you can run the `/iter_exp_bc.sh` and `iter_exp_hg.sh`, which apply our 2 fine-tuned language models `/partBC_30epochs` and `/partHG_1epoch`, respectively. Also don't forget to customize the GPU resource related parameters in the scripts.

**Run your own expriments**  
(Not finished)
