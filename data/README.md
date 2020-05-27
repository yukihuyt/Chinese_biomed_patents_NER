# Dataset Information
## Labeled dataset
The derictory `cbp_gold` contains our finally built gold standard dataset (humanly labeled), which are `cbp_gold_total.bio` (original version) and `no_long_cbp_gold_total.bio` (no-long-sentences version). This gold standard dataset contains 5,813 sentences and 2,267 unique named entities, built from 21 Chinese biomedical patents.
It was annotated with IOB format labels and we only annotated out gene/protein/disease entities. In the data document, each line only contains one character and its corresponding IOB tag.

Under the directory `cbp_gold`, there are also 5 sub-directories (from `0` to `4`), which contain the detailed evaluation sets we built from our original gold standard dataset and were finally applied in all our NER evaluation experiments. Under each sub-directory, the filename suffix indicates whether it is train, test or dev set, while the prefix indicates whether it contains long sentences. (file with prefix `no_long` does not contain sentences longer than 500 characters, exceeded length sentences will be split into sub-sentences.)

## Unlabeled datasets
`BC` and `HG` are two unlabeled Chinese Biomedical Patents datasets, which were both retrieved from Google Patents Databases. `BC` contains patents matching the query "人类AND基因" ('human AND gene'), from 1st January 2009 to 1st January 2019 with patent code starting with 'CN'. `HG` contains patents matching the query  "乳腺癌AND生物标记物" ('breast cancer AND biomarker'), from 1st December 2012 to 1st January 2019 with patent code starting with 'CN' as well. After some filtering and cleaning steps (detailed steps see our paper), `BC` and `HG` datasets finally contain 2,659 and 53,007 patents, respectively.

The `part_BC` and `part_HG` were part of both datasets, which contain 100 and 10,000 patents randomly selected from `BC` and `HG` datasets, respectively. They were processed as the suggested format for the transformers frame we applied and were finally used in our Language Model fine-tuning experiments (Mixed_LM).

Here we would not upload these unlabeled datasets since they are all relatively large and are not necessary if you just want to reproduce our NER results or use our trained models. 

Detailed statistics of all our built dataset can be found in our paper.  


 