import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import numpy as np

from flask_app import DEVICE
from hp import ALPHA
from data_load import TAG_TO_INDEX, INDEX_TO_TAG, NUM_TASK


def eval(model, iterator, f, criterion, binary_criterion, baseline_1=False, NUM_TASK=NUM_TASK):
    """ evaluation on SLC and FLC tasks """

    VOCAB = [
        ("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans",
         "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation",
         "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon",
         "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion",
         "Appeal_to_Authority", "Black-and-White_Fallacy",
         "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism")
    ]

    if NUM_TASK == 2:
        VOCAB.append(("Non-prop", "Prop"))

    for i in range(NUM_TASK):
        TAG_TO_INDEX.append({tag: idx for idx, tag in enumerate(VOCAB[i])})
        INDEX_TO_TAG.append({idx: tag for idx, tag in enumerate(VOCAB[i])})

    model.eval()

    valid_losses = []

    Words, Is_heads = [], []
    Tags = [[] for _ in range(NUM_TASK)]
    Y = [[] for _ in range(NUM_TASK)]
    Y_hats = [[] for _ in range(NUM_TASK)]
    with torch.no_grad():
        for batch in iterator:
            words, x, is_heads, att_mask, tags, y, seqlens = batch
            att_mask = torch.Tensor(att_mask)
            logits, y_hats = model(x, attention_mask=att_mask)  # logits: (N, T, VOCAB), y: (N, T)

            loss = []
            for i in range(NUM_TASK):
                logits[i] = logits[i].view(-1, logits[i].shape[-1])  # (N * T, 2)
            y[0] = y[0].view(-1).to(DEVICE)
            loss.append(criterion(logits[0], y[0]))
            if NUM_TASK == 2:
                y[1] = y[1].float().to(DEVICE)
                loss.append(binary_criterion(logits[1], y[1]))

            if NUM_TASK == 1:
                joint_loss = loss[0]
            elif NUM_TASK == 2:
                joint_loss = ALPHA * loss[0] + (1 - ALPHA) * loss[1]

            valid_losses.append(joint_loss.item())
            Words.extend(words)
            Is_heads.extend(is_heads)

            for i in range(NUM_TASK):
                Tags[i].extend(tags[i])
                Y[i].extend(y[i].cpu().numpy().tolist())
                Y_hats[i].extend(y_hats[i].cpu().numpy().tolist())
    valid_loss = np.average(valid_losses)

    with open(f, 'w', encoding='utf-8') as fout:
        y_hats, preds = [[[] for _ in range(NUM_TASK)] for _ in range(2)]
        if NUM_TASK == 1:
            for words, is_heads, tags[0], y_hats[0] in zip(Words, Is_heads, *Tags, *Y_hats):
                y_hats[0] = [hat for head, hat in zip(is_heads, y_hats[0]) if head == 1]
                preds[0] = [INDEX_TO_TAG[0][hat] for hat in y_hats[0]]
                fout.write(words.split()[0])
                fout.write("\n")
                for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                    fout.write("{} {} {} \n".format(w, t1, p_1))
                fout.write("\n")
        else:  # NUM_TASK == 2
            TP, FP, FN, TN = 0, 0, 0, 0
            for words, is_heads, tags[0], tags[1], y_hats[0], y_hats[1] in zip(Words, Is_heads, *Tags, *Y_hats):
                y_hats[0] = [hat for head, hat in zip(is_heads, y_hats[0]) if head == 1]
                preds[0] = [INDEX_TO_TAG[0][hat] for hat in y_hats[0]]
                preds[1] = INDEX_TO_TAG[1][y_hats[1]]

                if baseline_1:
                    preds[1] = 'Prop'

                if tags[1].split()[1] == 'Non-prop' and preds[1] == 'Non-prop':
                    TN += 1
                elif tags[1].split()[1] == 'Non-prop' and preds[1] == 'Prop':
                    FP += 1
                elif tags[1].split()[1] == 'Prop' and preds[1] == 'Prop':
                    TP += 1
                elif tags[1].split()[1] == 'Prop' and preds[1] == 'Non-prop':
                    FN += 1

                fout.write(words.split()[0] + "\n")
                for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                    fout.write("{} {} {} {} {}\n".format(w, t1, tags[1].split()[1:-1][0], p_1, preds[1]))
                fout.write("\n")

            try:
                precision = TP / (TP + FP)
            except ZeroDivisionError:
                precision = 1.0
            try:
                recall = TP / (TP + FN)
            except ZeroDivisionError:
                recall = 1.0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                if precision * recall == 0:
                    f1 = 1.0
                else:
                    f1 = 0.0
            print(f"SLC precision: {precision:.4f}")
            print(f"SLC recall: {recall:.4f}")
            print(f"SLC f1-score: {f1:.4f}")

    ## calc metric
    y_true, y_pred = [], []
    for i in range(NUM_TASK):
        y_true.append(np.array([
            TAG_TO_INDEX[i][line.split()[i + 1]]
            for line in open(f, 'r', encoding='utf-8').read().splitlines()
            if len(line.split()) > 1
        ]))
        if baseline_1:
            if i == 0:
                key = "Loaded_Language"
            else:
                key = "Prop"
            y_pred.append(np.array([
                TAG_TO_INDEX[i][key]
                for line in open(f, 'r', encoding='utf-8').read().splitlines()
                if len(line.split()) > 1
            ]))
        else:
            y_pred.append(np.array([
                TAG_TO_INDEX[i][line.split()[i + 1 + NUM_TASK]]
                for line in open(f, 'r', encoding='utf-8').read().splitlines()
                if len(line.split()) > 1
            ]))

    num_predicted, num_correct, num_gold = 0, 0, 0

    num_predicted += len(y_pred[0][y_pred[0] > 1])
    num_correct += (np.logical_and(y_true[0] == y_pred[0], y_true[0] > 1)).astype(np.int).sum()
    num_gold += len(y_true[0][y_true[0] > 1])

    print(f"FLC number of predicted techniques: {num_predicted}")
    print(f"FLC number of correct techniques: {num_correct}")
    print(f"FLC number of gold techniques: {num_gold}")

    try:
        precision = num_correct / num_predicted
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    print(f"FLC precision: {precision:.4f}")
    print(f"FLC recall: {recall:.4f}")
    print(f"FLC f1-score: {f1:.4f}")
    return precision, recall, f1, valid_loss
