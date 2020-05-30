import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils import data

from hp import SEQ_LEN, BERT, JOINT, GRANU, MGN
from preprocess import make_dataset, make_bert_dataset

if BERT:
    NUM_TASK = 1
    MASKING = False
    HIER = False
elif JOINT:
    NUM_TASK = 2
    MASKING = False
    HIER = False
elif GRANU:
    NUM_TASK = 2
    MASKING = False
    HIER = True
elif MGN:
    NUM_TASK = 2
    MASKING = True
    HIER = False
else:
    raise ValueError("You should choose one of bert, joint, granu and mgn in options")

TAG_TO_INDEX, INDEX_TO_TAG = [], []

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

tokenizer = BertTokenizer.from_pretrained('bert/vocab.txt', do_lower_case=False)


class PropDataset(data.Dataset):
    def __init__(self, directory, is_test=False, verbose=False):
        dataset = make_dataset(directory)
        words, tags, ids = make_bert_dataset(dataset, is_test=is_test, verbose=verbose)
        # (
        #     [ [ ['first_word', 'second_word', ...], ... ], ... ],
        #     [ [ ['label1', 'label2', ...], ... ], ... ],
        #     [ [ id1, id2, ... ], ... ]
        # )

        flat_ids, flat_sents = [], []
        tags_li = [[] for _ in range(NUM_TASK)]
        for article_words, article_tags, article_ids in zip(words, tags, ids):
            for inner_words, inner_tags, id_ in zip(article_words, article_tags, article_ids):
                flat_sents.append(["[CLS]"] + inner_words + ["[SEP]"])
                flat_ids.append(id_)

                tmp_tags = []
                if NUM_TASK == 1:  # technique classification
                    tmp_tags.append(['O'] * len(inner_tags))
                    for j, inner_tag in enumerate(inner_tags):
                        if inner_tag != 'O' and inner_tag in VOCAB[0]:
                            tmp_tags[0][j] = inner_tag
                    tags_li[0].append(["<PAD>"] + tmp_tags[0] + ["<PAD>"])
                else:  # sentence classification
                    tmp_tags.append(['O'] * len(inner_tags))
                    tmp_tags.append(['Non-prop'])
                    for j, inner_tag in enumerate(inner_tags):
                        if inner_tag != 'O' and inner_tag in VOCAB[0]:
                            tmp_tags[0][j] = inner_tag
                            tmp_tags[1] = ['Prop']
                    for i in range(NUM_TASK):
                        tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])

        self.sents, self.ids, self.tags_li = flat_sents, flat_ids, tags_li
        assert len(self.sents) == len(self.ids) == len(self.tags_li[0])

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        words = self.sents[index]
        id_ = self.ids[index]
        tags = list(list(zip(*self.tags_li))[index])  # [ ['label1', 'label2', ...] ]

        x, is_heads = [], []  # list of ids
        y = [[] for _ in range(NUM_TASK)]  # list of lists of lists
        tt = [[] for _ in range(NUM_TASK)]  # list of lists of lists

        for word, tag in zip(words, tags[0]):
            tokens = tokenizer.tokenize(word) if word not in ("[CLS]", "[SEP]") else [word]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)
            if len(xx) < len(is_head):
                xx = xx + [100] * (len(is_head) - len(xx))  # 100 == "[UNK]"

            tag = [tag] + [tag] * (len(tokens) - 1)
            y[0].extend([TAG_TO_INDEX[0][each] for each in tag])
            tt[0].extend(tag)

            x.extend(xx)
            is_heads.extend(is_head)

        if NUM_TASK == 2:
            if tags[1][1] == 'Non-prop':
                y[1].extend([1, 0])
                tt[1].extend([tags[1][1]])
            elif tags[1][1] == 'Prop':
                y[1].extend([0, 1])
                tt[1].extend([tags[1][1]])

        seqlen = len(y[0])
        words = " ".join([id_] + words)

        for i in range(NUM_TASK):
            tags[i] = " ".join(tags[i])

        att_mask = [1] * seqlen
        return words, x, is_heads, att_mask, tags, y, seqlen


def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = SEQ_LEN

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: '[PAD]'
    x = torch.LongTensor(f(1, maxlen))
    att_mask = f(-4, maxlen)

    y = []
    tags = []

    y.append(torch.LongTensor([sample[-2][0] + [0] * (maxlen - len(sample[-2][0])) for sample in batch]))
    for i in range(NUM_TASK):
        tags.append([sample[-3][i] for sample in batch])
    if NUM_TASK == 2:
        y.append(torch.LongTensor([sample[-2][1] for sample in batch]))

    return words, x, is_heads, att_mask, tags, y, seqlen
