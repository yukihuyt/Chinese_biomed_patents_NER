import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, tokenizer
from train import eval_model
import os
import numpy as np
import argparse
import time
# import setproctitle
# setproctitle.setproctitle("demo_finetuning")

def read_doc(doc_filepath):
    doc_bychar=[]
    with open(doc_filepath, 'r', encoding='utf-8') as fin:
        for line in fin:
            line=line.strip()
            doc_bychar.append([f'{x} O' for x in line])
    return doc_bychar

def write_bio(doc_bychar, biopath):
    with open(biopath, 'w+', encoding='utf-8') as ft:
        out_contents=['\n'.join(y) for y in doc_bychar]
        ft.write('\n\n'.join(out_contents))

def pred(model, iterator):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    pred_contents=[]
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        pred_one_sent=[]
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        assert len(preds)==len(words.split())==len(tags.split())
        for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
            pred_one_sent.append(f"{w} {p}")
        pred_contents.append()

    return pred_contents
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--pred_dir", type=str, default="conll2003")
    parser.add_argument("--outpred_dir", type=str, default="checkpoints/01")
    parser.add_argument("--testset", type=str, default="")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--testing", dest="testing", action="store_true")
    parser.add_argument("--predicting", dest="testing", action="store_true")
    parser.add_argument("--load_weights", dest="load_weights", action="store_true")
    parser.add_argument("--model_state_path", type=str, default="")
    parser.add_argument("--vocab_path", type=str, default="")

    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    VOCAB=[]
    with open(hp.vocab_path, 'r', 'encoding=utf-8') as fv:
        for line in fv:
            line=line.strip()
            if line:
                VOCAB.append(line)
    print ("\nRead the vocab tags: {}".format(VOCAB))

    print ("\nParameter settings:\n {}\n".format(hp))
    tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
    idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}


    model = Net(hp.top_rnns, len(VOCAB), device, hp.finetuning, hp.rnn_layers).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(hp.model_state_path))

    if hp.testing:
        if hp.testset:
            print("=========Testing with selected model=========")
            test_dataset = NerDataset(hp.testset)
            test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=hp.test_batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad)
            fname_test = os.path.join(hp.logdir, "testing_with_selected")
            precision, recall, f1 = eval_model(model, test_iter, fname_test, detail_or_not=True, save_or_not=True)
        else:
            print("Error: no input test set filepath given")
    
    if hp.predicting:
        if hp.pred_dir:
            print("=========Making predictions with selected model=========")
            inputfiles=os.listdir(hp.pred_dir)
            for onefile in inputfiles:
                inputpath=os.path.join(hp.pred_dir, onefile)
                temppath=os.path.join(hp.logdir, onefile.replace('.txt', '.temp'))
                outpath=os.path.join(hp.outpred_dir, onefile.replace('.txt', '.pred'))

                input_doc=read_doc(inputpath)
                write_bio(input_doc, temppath)
                
                pred_dataset=NerDataset(temppath)
                pred_iter = data.DataLoader(dataset=pred_dataset,
                                    batch_size=hp.test_batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad)
                
                pred_doc=pred(model, pred_iter)
                write_bio(outpath, pred_doc)
                
                os.remove(temppath)
            print(f"Made predictions on {len(inputfiles)} files.\n")
                
        else:
            print("Error: no input pred set dirpath given")
            

        
        

