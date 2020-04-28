export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased

python3 preprocess.py ./data/NER-de-dev.txt $BERT_MODEL $MAX_LENGTH > ./data/train.txt
python3 preprocess.py ./data/NER-de-dev.txt $BERT_MODEL $MAX_LENGTH > ./data/dev.txt
python3 preprocess.py ./data/NER-de-dev.txt $BERT_MODEL $MAX_LENGTH > ./data/test.txt