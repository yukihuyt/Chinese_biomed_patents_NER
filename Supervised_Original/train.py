import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, tokenizer
import os
import numpy as np
import argparse
import time
from seqeval.metrics import classification_report
# import setproctitle
# setproctitle.setproctitle("demo_finetuning")


def train(model, iterator, optimizer, criterion, sanity_check=False):
    model.train()
    print("=======================")
    print("Training started at {}".format(time.asctime(time.localtime(time.time()))))
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if sanity_check:
            if i==0:
                print("=====sanity check======")
                print("words:", words[0])
                print("x:", x.cpu().numpy()[0][:seqlens[0]])
                print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
                print("is_heads:", is_heads[0])
                print("y:", _y.cpu().numpy()[0][:seqlens[0]])
                print("tags:", tags[0])
                print("seqlen:", seqlens[0])
                print("=======================")
            
        if i%100==0: # monitoring
            print(f"time: {time.asctime(time.localtime(time.time()))}, step: {i}, loss: {loss.item()}")

    print(f"Finished {i} steps training at {time.asctime(time.localtime(time.time()))}\n")
    return loss.item()

# def cal_metrics(labels_pred, labels_true, tags_tp, tags_ignore=["O", "<PAD>"]):


def eval_model(model, iterator, f, save_or_not=False, detail_or_not=False):
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
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    label_true = [line.split()[1] for line in open("temp", 'r').read().splitlines() if len(line) > 0]
    label_pred = [line.split()[2] for line in open("temp", 'r').read().splitlines() if len(line) > 0]
    tags_ignore = ["O", "<PAD>"]

    ###########for each type of label##########
    if detail_or_not:
        real_pred = [x for x in label_pred if x !="<PAD>"]
        real_true = [x for x in label_true if x !="<PAD>"]
        cls_report=classification_report(real_true, real_pred)
        print (cls_report)
        # label_types = []

        # for tag in tag2idx.keys():
        #     if tag not in tags_ignore and tag[-3:] not in label_types:
        #         label_types.append(tag[-3:])
        
        # detailed_eval = {}

        # for one_type in label_types:
        #     num_tp=0
        #     num_fp=0
        #     num_tn=0
        #     num_fn=0

        #     for i, one_label in enumerate(label_true):
        #         if len(label_true[i])>3 and label_true!="<PAD>":
        #             if label_true[i]==label_pred[i]:
        #                 if label_pred[i][-3:]==one_type:
        #                     num_tp+=1
        #                 else:
        #                     num_tn+=1
        #             else:
        #                 if label_pred[i][-3:]==one_type:
        #                     num_fp+=1
        #                 else:
        #                     num_fn+=1

        #     str_d01 = f"Eval results of {one_type}: tp: {num_tp}, tn: {num_tn}, fp: {num_fp}, fn: {num_fn}"
            
        #     try:
        #         this_pre = float(num_tp)/float(num_tp+num_fp)
        #     except ZeroDivisionError:
        #         this_pre=1.0000
            
        #     try:
        #         this_rec = float(num_tp)/float(num_tp+num_fn)
        #     except ZeroDivisionError:
        #         this_rec=1.0000
        #     try:
        #         this_f1 = 2.0000*this_pre*this_rec/(this_pre+this_rec)
        #     except ZeroDivisionError:
        #         if this_pre*this_rec==0:
        #             this_f1=1.0000
        #         else:
        #             this_f1=0.0000
            
        #     str_d02 = f"Precision: {this_pre:.4f}, Recall: {this_rec:.4f}, F1: {this_f1:.4f}"
            
        #     detailed_eval[one_type]=str_d01+"\n"+str_d02
        #     print(str_d01+"\n"+str_d02)

        ################################################################################

    num_total_example = len(label_pred)
    num_total_tp_fp = len([x for x in label_pred if x not in tags_ignore])
    num_total_tp_fn = len([x for x in label_true if x not in tags_ignore])
    num_total_tp = 0
    for i, one_label in enumerate(label_pred):
        if label_pred[i]==label_true[i] and one_label not in tags_ignore:
            num_total_tp+=1

    # num_proposed = len(y_pred[y_pred>1])
    # num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    # num_gold = len(y_true[y_true>1])

    print("Micro avg results:")
    print(f"total number of example:{num_total_example}")
    print(f"total tp:{num_total_tp}")
    print(f"total proposed (tp+fp):{num_total_tp_fp}")
    print(f"total positive (tp+fn):{num_total_tp_fn}")
    try:
        precision = num_total_tp / num_total_tp_fp
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_total_tp / num_total_tp_fn
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    if save_or_not:
        final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
        with open(final, 'w') as fout:
            result = open("temp", "r").read()
            fout.write("word label prediction\n")
            fout.write(f"{result}\n")
            
            if detail_or_not:
                fout.write("---------------------------------------------------\n")
                fout.write("Detailed eval results:\n")
                fout.write(f"{cls_report}\n")
            
            fout.write("---------------------------------------------------\n")
            fout.write("Micro average results:\n")
            fout.write(f"total_example:{num_total_example}, total_tp:{num_total_tp}, total tp+fp:{num_total_tp_fp}, total tp+fn:{num_total_tp_fn}\n")
            fout.write(f"precision={precision:.4f}\n")
            fout.write(f"recall={recall:.4f}\n")
            fout.write(f"f1={f1:.4f}\n")

    os.remove("temp")

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    return precision, recall, f1

if __name__=="__main__":
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
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    parser.add_argument("--testset", type=str, default="conll2003/test.txt")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--testing", dest="testing", action="store_true")
    parser.add_argument("--load_weights", dest="load_weights", action="store_true")
    parser.add_argument("--weights_path", type=str, default="")


    hp = parser.parse_args()

    if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = NerDataset(hp.trainset)
    eval_dataset = NerDataset(hp.validset)
    
    VOCAB=train_dataset.VOCAB
    print ("\nGet the following tags in trainset: {}".format(VOCAB[1:]))

    vocab_path=os.path.join(hp.logdir, "VOCAB.txt")
    with open(vocab_path, 'w+', encoding='utf-8') as f:
        f.write('\n'.join(VOCAB))

    print ("\nParameter settings:\n {}\n".format(hp))
    tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
    idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    model = Net(hp.top_rnns, len(VOCAB), device, hp.finetuning, hp.rnn_layers).cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr, weight_decay=hp.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = 100000
    best_epoch = 0

    for epoch in range(1, hp.n_epochs+1):
        
        fname_temp = os.path.join(hp.logdir, "temp"+str(epoch))

        if epoch==1:
            loss_this_epoch = train(model, train_iter, optimizer, criterion, sanity_check=True)
        else:
            loss_this_epoch = train(model, train_iter, optimizer, criterion)

        print(f"=========eval at epoch={epoch}=========")
        if epoch%5==0:
            precision, recall, f1 = eval_model(model, eval_iter, fname_temp, save_or_not=True)

        else:
            precision, recall, f1 = eval_model(model, eval_iter, fname_temp)

        if best_loss>loss_this_epoch:
            best_model_state = model.state_dict()
            best_loss = loss_this_epoch
            best_epoch = epoch

        if epoch%20==0 and epoch!=hp.n_epochs:
            torch.save(model.state_dict(), f"{fname_temp}.pt")
            print(f"weights were saved to {fname_temp}.pt")

        if epoch==hp.n_epochs:
            fname_best = os.path.join(hp.logdir, 'best_model')
            torch.save(best_model_state, f"{fname_best}.pt")
            fname_final = os.path.join(hp.logdir, 'last_model')
            torch.save(model.state_dict(), f"{fname_final}.pt")

    if hp.testing:
        print(f"=========Testing on best model=========")
        test_dataset = NerDataset(hp.testset)
        test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.test_batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
        fname_test = os.path.join(hp.logdir, "test_on_best_"+str(best_epoch))
        model.load_state_dict(best_model_state)
        precision, recall, f1 = eval_model(model, test_iter, fname_test, detail_or_not=True, save_or_not=True)
        print("=======================\n")
