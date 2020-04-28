import os
import argparse
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, BytePairEmbeddings
from typing import List
from flair.embeddings import BertEmbeddings
from torch.optim.adam import Adam
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default=None, type=str, required=True, 
                      help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
  parser.add_argument("--train_filename", default=None, type=str, required=True, help="train data filename")
  parser.add_argument("--test_filename", default=None, type=str, required=True, help="test data filename")
  parser.add_argument("--dev_filename", default=None, type=str, required=True, help="dev data filename")
  parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output directory where the model predictions and checkpoints will be written.")
  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument("--learning_rate", default=5e-5, type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs", default=3, type=int,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--hidden_layer_size", default=128, type=int,
                      help="Hidden layer size.")
  
  args = parser.parse_args()

  # define columns
  columns = {0: 'text', 1: 'ner'}

  # this is the folder in which train, test and dev files reside
  data_folder = args.data_dir

  # retrieve corpus using column format, data folder and the names of the train, dev and test files
  corpus: Corpus = ColumnCorpus(data_folder, columns,
    train_file=args.train_filename,
    test_file=args.test_filename,
    dev_file=args.dev_filename)

  # 2. what tag do we want to predict?
  tag_type = 'ner'

  # 3. make the tag dictionary from the corpus
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  print(tag_dictionary.idx2item)

  print (len(corpus.train))

  # 5. initialize sequence tagger
  bert_default = BertEmbeddings('bert-base-chinese')
  # bert_full=BertEmbeddings(
  #   bert_model_or_path='bert-base-chinese',
  #   layers="0,1,2,3,4,5,6,7,8,9,10,11,12",
  #   pooling_operation="first",
  #   use_scalar_mix=False
  #   )

  # 5. initialize sequence tagger
  tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_layer_size,
                                          embeddings=bert_default,
                                          tag_dictionary=tag_dictionary,
                                          tag_type=tag_type,
                                          use_crf=True,
                                          use_rnn=True)

  # 6. initialize trainer
  
  trainer: ModelTrainer = ModelTrainer(tagger, corpus)

  # 7. start training
  if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
  trainer.train(args.output_dir,
                learning_rate=args.learning_rate,
                mini_batch_size=args.per_gpu_train_batch_size,
                max_epochs=args.num_train_epochs)

  # 8. plot weight traces (optional)
  # from flair.visual.training_curves import Plotter
  # plotter = Plotter()
  # plotter.plot_weights('resources/taggers/cbp_demo00/weights.txt')