from pathlib import Path

import numpy as np


def read_data(directory, is_test=False):
    """
    returns: (['id1', 'id2', ...], ['text1', 'text2', ...], [[[left, right, technique, intersection, more_than_sent], ...], ...])
    """

    ids = []
    texts = []
    if not is_test:
        labels = []
    for f in directory.glob('*.txt'):
        ids.append(f.name.replace('article', '').replace('.txt', ''))
        texts.append(f.read_text(encoding='utf-8'))
        if not is_test:
            labels.append(parse_labels(f.as_posix().replace('.txt', '.labels.tsv')))
    if not is_test:
        return ids, texts, labels
    return ids, texts


def parse_labels(label_path):
    """
    returns: [[left, right, technique, intersection, more_than_sent], ...]
    """

    labels = []
    if not Path(label_path).exists():
        return labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            _, technique, left, right = line.strip().split('\t')
            labels.append([int(left), int(right), technique, 0, 0])
    labels.sort()
    if not labels:
        return labels
    length = max([label[1] for label in labels])
    visited = np.zeros(length)
    for label in labels:
        if sum(visited[label[0]:label[1]]):
            label[3] = 1  # intersection
        else:
            visited[label[0]:label[1]] = 1
    return labels


def clean_text(articles, ids):
    """
    articles: ['first text here', 'second text here', ...]
    ids: ['id1', 'id2', ...]
    returns: [[[id_, sentence, start, end], ...], ...]
    """

    texts = []
    for article, id_ in zip(articles, ids):
        sentences = article.split('\n')  # ['first sentence', 'second sentence', ...]
        end = -1
        res = []
        for sentence in sentences:
            start = end + 1
            end = start + len(sentence)  # length of sequence
            if sentence:
                res.append([id_, sentence, start, end])  # [[id_, sentence, start, end], ...]
        texts.append(res)
    return texts  # [[[id_, sentence, start, end], ...], ...]


def make_dataset(directory):
    """
    returns: [[['id1', 'sentence1', start, end, left, right, technique, intersection, more_than_sent], ...], ...]
    """

    ids, texts, group_labels = read_data(directory)
    # ids: ['id1', 'id2', ...]
    # texts: ['text1', 'text2', ...]
    # group_labels: [[[left, right, technique, intersection, more_than_sent], ...], ...]
    texts = clean_text(texts, ids)
    # texts: [[['id1', 'sentence1', start, end], ['id1', 'sentence2', start, end], ...], ...]
    res = []
    for sents, labels in zip(texts, group_labels):
        # sents: [['id1', 'sentence1', start, end], ['id1', 'sentence2', start, end], ...]
        # labels: [[left, right, technique, intersection, more_than_sent], ...]

        # making positive examples
        tmp = []
        pos_ind = [0] * len(sents)
        for label in labels:
            # label: [left, right, technique, intersection, more_than_sent]
            left, right, technique, intersection, more_than_sent = label
            for i, sent in enumerate(sents):
                # sent: ['id1', 'sentence1', start, end]
                *_, start, end = sent
                if left >= start and left < end and right > end:
                    label[4] = 1
                    tmp.append(sent + [left, end, technique, intersection, label[4]])
                    pos_ind[i] = 1
                    label[0] = end + 1
                elif left != right and left >= start and left < end and right <= end:
                    tmp.append(sent + label)
                    # tmp: [['id1', 'sentence1', start, end, left, right, technique, intersection, more_than_sent], ...]
                    pos_ind[i] = 1

        # making negative examples
        dummy = [0, 0, 'O', 0, 0]
        for i, sent in enumerate(sents):
            if pos_ind[i] != 1:
                tmp.append(sent + dummy)
        res.append(tmp)
        # res: [[['id1', 'sentence1', start, end, left, right, technique, intersection, more_than_sent], ...], ...]
    return res


def make_bert_dataset(dataset, is_test=False, verbose=False):
    """
    dataset: [[['id1', 'sentence1', start, end, left, right, technique, intersection, more_than_sent], ...], ...]
    returns: (
                [ [ ['first_word', 'second_word', ...], ... ], ... ],
                [ [ ['label1', 'label2', ...], ... ], ... ],
                [ [ id1, id2, ... ], ... ]
            )
    """

    words, tags, ids = [], [], []
    for article in dataset:
        # article: [['id1', 'sentence1', start, end, left, right, technique, intersection, more_than_sent], ...]
        tmp_doc, tmp_label, tmp_id = [], [], []
        tmp_sen = article[0][1]
        tmp_i = article[0][0]
        label = ['O'] * len(tmp_sen.split(' '))
        for sentence in article:
            # sentence: ['id1', 'sentence1', start, end, left, right, technique, intersection, more_than_sent]
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9:  # label exists
                if tmp_sen != sentence[1] or (sentence[7] and is_test):
                    tmp_label.append(label)
                    tmp_doc.append(tmp_sen.split(' '))
                    tmp_id.append(tmp_i)
                    if tmp_sen != sentence[1]:
                        label = ['O'] * len(token_len)
                start = sentence[4] - sentence[2]
                end = sentence[5] - sentence[2]
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)):
                        token_len[i] += token_len[i - 1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end)
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else:
                        e_ind = s_ind
                    for i in range(s_ind, e_ind + 1):
                        label[i] = sentence[6]
                tmp_sen = sentence[1]
                tmp_i = sentence[0]
            else:
                tmp_doc.append(tokens)
                tmp_id.append(sentence[0])
        if len(sentence) == 9:
            tmp_label.append(label)
            tmp_doc.append(tmp_sen.split(' '))
            tmp_id.append(tmp_i)
        words.append(tmp_doc)
        tags.append(tmp_label)
        ids.append(tmp_id)
    if verbose:
        print(f'words: {words}')
        print(f'tags: {tags}')
        print(f'ids: {ids}')
    return words, tags, ids
