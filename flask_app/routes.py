import random as random_module
import sys
from collections import deque, defaultdict
from functools import lru_cache
from pathlib import Path

import nltk
from flask import render_template, request, jsonify, send_from_directory
from torch.utils import data

from data_load import PropDataset, pad
from eval.convert import convert, remove_duplicates
from flask_app import app, criterion, binary_criterion
from hp import BATCH_SIZE, BERT_PATH, JOINT_BERT_PATH, GRANU_BERT_PATH, MGN_SIGM_BERT_PATH
from preprocess import read_data, clean_text
from settings import load_model
from train import eval

sys.path.append('../secret_repo')

BASE_DIR = Path.cwd()
DIRECTORY_TRAIN = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'train')
DIRECTORY_DEV = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'dev')
DIRECTORY_TEST = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'test')
DIRECTORY_MARKUP = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'markup')
DIRECTORY_PREDICT = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'predict')

TECHNIQUES = [
    'No', 'Whataboutism', 'Thought-terminating_Cliches', 'Straw_Men', 'Slogans', 'Repetition',
    'Reductio_ad_hitlerum', 'Red_Herring', 'Obfuscation,Intentional_Vagueness,Confusion',
    'Name_Calling,Labeling', 'Loaded_Language', 'Flag-Waving', 'Exaggeration,Minimisation',
    'Doubt', 'Causal_Oversimplification', 'Black-and-White_Fallacy', 'Bandwagon',
    'Appeal_to_fear-prejudice', 'Appeal_to_Authority'
]
HUMAN_READABLE_TECHNIQUES = [
    "No", "Whataboutism", "Thought-terminating Cliches", "Straw Men", "Slogans", "Repetition",
    "Reductio ad hitlerum", "Red Herring", "Obfuscation, Intentional Vagueness, Confusion",
    "Name Calling, Labeling", "Loaded Language", "Flag-Waving", "Exaggeration, Minimisation",
    "Doubt", "Causal Oversimplification", "Black-and-White Fallacy", "Bandwagon",
    "Appeal to fear-prejudice", "Appeal to Authority"
]
TYPE_TO_DIRECTORY = {
    'train': DIRECTORY_TRAIN,
    'dev': DIRECTORY_DEV,
    'test': DIRECTORY_TEST,
    'markup': DIRECTORY_MARKUP,
}
N = 10e10
ARTICLE = 7
TYPE_TEST = '_test'


@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    text = request.form.get('text') or ''
    return render_template('index.html', text=text)


@app.route('/static/<path:path>')
def send_pictures(path):
    return send_from_directory('static', path)


@app.route('/markup', methods=['GET'])
def markup():
    return render_template('markup.html')


@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')


@app.route('/random', methods=['GET'])
def random():
    return render_template('random.html')


@app.route('/articles', methods=['GET'])
def articles():
    articles = []
    for directory, type_ in zip(
            [DIRECTORY_TRAIN, DIRECTORY_DEV, DIRECTORY_TEST, DIRECTORY_MARKUP, DIRECTORY_PREDICT],
            ['train', 'dev', 'test', 'markup', 'predict']):
        for f in directory.glob('*.txt'):
            id_ = int(f.name.split('.')[0][7:])
            text = f.read_text(encoding='utf-8')
            articles.append({
                'id': id_,
                'title': text.split('\n')[0],
                'size': len(text),
                'type': type_,
                'markuper': 'egrigorev',  # NB: replace later
            })
    return render_template('articles.html', articles=articles)


@app.route('/articles/<article_id>', methods=['GET'])
def article(article_id):
    article_type = request.args.get('article_type')
    article_title = request.args.get('article_title')
    directory = TYPE_TO_DIRECTORY[article_type]
    text = directory.joinpath(f'article{article_id}.txt').read_text(encoding='utf-8')
    correct_lst = _get_correct_list(article_id, directory=directory)
    return render_template('article.html', list=correct_lst, text=text, title=article_title)


def get_existent_ids(directories=(
        DIRECTORY_TRAIN, DIRECTORY_DEV, DIRECTORY_TEST, DIRECTORY_MARKUP, DIRECTORY_PREDICT)):
    ids = set()
    for directory in directories:
        for f in directory.glob('*.txt'):
            ids.add(int(f.name.split('.')[0][7:]))
    return ids


def get_existent_dicts():
    dct = {}
    for filename in DIRECTORY_MARKUP.glob('*.tsv'):
        id_ = int(str(filename.parts[-1]).split('.')[0][7:])
        dct[id_] = get_list(id_)
    return dct


def get_list(id_, directory=DIRECTORY_MARKUP):
    """
    Функция, возвращающая соответствующий список, если id в списке id-шников,
    в противном случае выкидывающая ошибку.
    """

    if id_ not in get_existent_ids():
        raise ValueError(f'No id {id_}')
    lines = []
    labels_file = directory.joinpath(f'article{id_}.labels.tsv')
    if labels_file.is_file():
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    with open(directory.joinpath(f'article{id_}.txt'), 'r', encoding='utf-8') as inner_f:
        length = len(inner_f.read())
    lst = [set() for _ in range(length)]
    for line in lines:
        id_, technique, left, right = line.split()
        id_, left, right = list(map(int, (id_, left, right)))
        for i in range(left, right):
            lst[i].add(technique)
    return lst


def write_existent_dict(id_, lst, directory=DIRECTORY_MARKUP):
    with open(directory.joinpath(f'article{id_}.labels.tsv'), 'w', encoding='utf-8') as f:
        queue = deque()
        res = []
        for i, elem in enumerate(lst):
            if elem:
                to_delete = []
                for j, queue_elem in enumerate(queue):
                    if queue_elem[0] not in elem:
                        queue[j][2] = i
                        res.append(queue[j])
                        to_delete.append(j)
                    else:
                        elem -= {queue_elem[0]}
                for inner_elem in elem:
                    queue.append([inner_elem, i, -1])
                for del_ix in to_delete[::-1]:
                    del queue[del_ix]
            else:
                for j, queue_elem in enumerate(queue):
                    queue[j][2] = i
                    res.append(queue[j])
                queue = deque()
        if queue:
            for j, queue_elem in enumerate(queue):
                queue[j][2] = i
                res.append(queue[j])
        for i in range(len(res)):
            res[i].insert(0, str(id_))
        res = [list(map(str, elem)) for elem in res]
        f.write('\n'.join(['\t'.join(elem) for elem in res]))


@app.route('/_add_technique', methods=['POST'])
def add_technique():
    full_text = request.form['full_text']
    left = int(request.form['left'])
    right = int(request.form['right'])
    id_ = request.form['id']
    # type_ = request.form['type']
    technique = TECHNIQUES[int(request.form['value'])]
    directory = DIRECTORY_MARKUP
    if not id_:
        ids = get_existent_ids()
        id_ = random_module.randint(0, N)
        while id_ in ids:
            id_ = random_module.randint(0, N)
        with open(directory.joinpath(f'article{id_}.txt'), 'w', encoding='utf-8') as f:
            f.write(full_text)
        directory.joinpath(f'article{id_}.labels.tsv').touch()
    else:
        id_ = int(id_)
    lst = get_list(id_, directory=directory)
    for i in range(left, right):
        if technique == TECHNIQUES[0]:
            lst[i] = set()
        else:
            lst[i].add(technique)
    write_existent_dict(id_, lst, directory=directory)
    correct_lst = _get_correct_list(id_, directory=directory)
    return jsonify(result={'id': id_, 'list': correct_lst})


def _get_correct_list(id_, directory=DIRECTORY_MARKUP):
    lst = get_list(int(id_), directory=directory)
    lst = [list(elem) for elem in lst]
    correct_lst = []
    for inner_lst in lst:
        techniques = []
        for elem in inner_lst:
            techniques.append(HUMAN_READABLE_TECHNIQUES[TECHNIQUES.index(elem)])
        correct_lst.append('; '.join(techniques))
    return correct_lst


def overwrite_one_article(id_, directory=DIRECTORY_PREDICT):
    lst = get_list(id_, directory=directory)
    symbols = []
    techniques = []
    with open(directory.joinpath(f'article{id_}.txt'), 'r+', encoding='utf-8') as f:
        text = f.read()
        sent_text = '\n'.join(nltk.sent_tokenize(text))  # , language="russian"
        i = 0
        j = 0
        while i < len(text) and j < len(sent_text):
            if text[i] == sent_text[j]:
                symbols.append(text[i])
                techniques.append(lst[i])
                i += 1
                j += 1
            else:
                if sent_text[j] == '\n':
                    symbols.append(sent_text[j])
                    techniques.append(lst[i])
                    j += 1
                while i < len(text) and j < len(sent_text) and text[i] != sent_text[j]:
                    i += 1
        f.seek(0)
        f.write(''.join(symbols))
        f.truncate()
    assert len(symbols) == len(techniques)
    # write_existent_dict(id_, techniques)
    return sent_text


@app.route('/_launch_model', methods=['POST'])
def launch_model():
    full_text = request.form['full_text']
    id_ = request.form['id']
    model_type = request.form['model_type']

    global BERT, JOINT, GRANU, MGN, NUM_TASK, MASKING, HIER
    BERT = model_type == BERT_PATH
    JOINT = model_type == JOINT_BERT_PATH
    GRANU = model_type == GRANU_BERT_PATH
    MGN = model_type == MGN_SIGM_BERT_PATH

    # either of the four variants:
    # BERT = False
    # JOINT = False
    # GRANU = False
    # MGN = True

    assert BERT or JOINT or GRANU or MGN
    assert not (BERT and JOINT) and not (BERT and GRANU) and not (BERT and MGN) \
           and not (JOINT and GRANU) and not (JOINT and MGN) and not (GRANU and MGN)

    # either of the two variants
    SIGMOID_ACTIVATION = True
    RELU_ACTIVATION = False
    assert not (SIGMOID_ACTIVATION and RELU_ACTIVATION) and (SIGMOID_ACTIVATION or RELU_ACTIVATION)

    if BERT:
        NUM_TASK = 1
        MASKING = 0
        HIER = 0
    elif JOINT:
        NUM_TASK = 2
        MASKING = 0
        HIER = 0
    elif GRANU:
        NUM_TASK = 2
        MASKING = 0
        HIER = 1
    elif MGN:
        NUM_TASK = 2
        MASKING = 1
        HIER = 0
    else:
        raise ValueError("You should choose one of bert, joint, granu and mgn in options")

    dct = {
        'NUM_TASK': NUM_TASK, 'MASKING': MASKING, 'SIGMOID_ACTIVATION': SIGMOID_ACTIVATION,
        'HIER': HIER
    }
    model = load_model(model_type, **dct)

    print(1)
    if not id_:
        print(2)
        ids = get_existent_ids()
        print(3)
        id_ = random_module.randint(0, N)
        print(4)
        while id_ in ids:
            id_ = random_module.randint(0, N)
        print(5)
        with open(DIRECTORY_PREDICT.joinpath(f'article{id_}.txt'), 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(6)
    print(7)
    text = overwrite_one_article(id_, directory=DIRECTORY_PREDICT)
    print(8)

    my_predict_dataset = PropDataset(DIRECTORY_PREDICT, is_test=True)
    print(9)
    my_predict_iter = data.DataLoader(
        dataset=my_predict_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
        collate_fn=pad
    )
    print(10)

    tmp_file = 'tmp.txt'
    print(11)
    eval(model, my_predict_iter, tmp_file, criterion, binary_criterion, NUM_TASK=NUM_TASK)
    print(12)
    ids, texts = read_data(DIRECTORY_PREDICT, is_test=True)
    print(13)
    t_texts = clean_text(texts, ids)
    print(14)
    flat_texts = [sentence for article in t_texts for sentence in article]
    print(15)
    fi, prop_sents = convert(NUM_TASK - 1, flat_texts, tmp_file)
    print(16)
    prop_sents = prop_sents[id_]
    print(17)
    prop_sents = ['1' if elem else '' for elem in prop_sents]
    print(18)
    results = remove_duplicates(fi)
    print(19)

    DIRECTORY_PREDICT.joinpath(f'article{id_}.txt').rename(
        DIRECTORY_MARKUP.joinpath(f'article{id_}.txt'))
    print(20)

    lst = [set() for _ in range(len(full_text))]
    print(21)
    source_lst = [set() for _ in range(len(full_text))]
    print(22)
    for inner_lst in results:
        for i in range(inner_lst[-2], inner_lst[-1]):
            lst[i].add(HUMAN_READABLE_TECHNIQUES[TECHNIQUES.index(inner_lst[-3])])
            source_lst[i].add(inner_lst[-3])
    print(23)

    correct_lst = ['; '.join(list(elem)) for elem in lst]
    print(24)
    write_existent_dict(id_, source_lst, directory=DIRECTORY_MARKUP)
    print(25)

    return jsonify(result={'id': id_, 'list': correct_lst, 'text': text, 'prop_sents': prop_sents})


def get_num_of_techniques_for_id(id_, directory=DIRECTORY_TRAIN):
    """
    Функция, возвращающая словарь с количеством употреблённых техник.
    """

    lines = []
    labels_file = directory.joinpath(f'article{id_}.labels.tsv')
    if labels_file.is_file():
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    label_count_dct = defaultdict(int)
    lst = []
    for line in lines:
        _, technique, left, right = line.split()
        left, right = list(map(int, [left, right]))
        technique = HUMAN_READABLE_TECHNIQUES[TECHNIQUES.index(technique)]
        label_count_dct[technique] += 1
        lst.append((technique, left, right))
    return label_count_dct, lst


@lru_cache(maxsize=-1, typed=False)
def get_id_dicts():
    id_to_text = {}
    id_to_labels = {}
    id_to_label_count = {}
    id_to_label_left_right = {}
    for directory in (DIRECTORY_TRAIN, DIRECTORY_DEV, DIRECTORY_TEST,
                      DIRECTORY_MARKUP, DIRECTORY_PREDICT):
        for f in directory.glob('*.txt'):
            id_ = int(f.name.split('.')[0][ARTICLE:])
            id_to_text[id_] = f.read_text(encoding='utf-8')
            id_to_labels[id_] = get_list(id_, directory=directory)
            id_to_label_count[id_], id_to_label_left_right[id_] = \
                get_num_of_techniques_for_id(id_, directory=directory)
    return id_to_text, id_to_labels, id_to_label_count, id_to_label_left_right


@lru_cache(maxsize=-1, typed=False)
def get_technique_to_examples():
    technique_to_examples = defaultdict(list)
    id_to_text, _, _, id_to_label_left_right = get_id_dicts()
    for id_, triples in id_to_label_left_right.items():
        text = id_to_text[id_]
        for triple in triples:
            label, left, right = triple
            sent_left, sent_right = left, right
            while sent_left >= 0 and text[sent_left] != '\n':
                sent_left -= 1
            while sent_right < len(text) and text[sent_right] != '\n':
                sent_right += 1
            sent_left += 1
            technique_to_examples[label].append((
                id_, text[left:right], text[sent_left:sent_right],
                left - sent_left, right - sent_left, label))
    return technique_to_examples


def get_random_technique():
    technique_to_examples = get_technique_to_examples()
    full_lst = [elem for technique in technique_to_examples for elem in technique_to_examples[technique]]
    return random_module.choice(full_lst)


@app.route('/_get_random_model', methods=['GET'])
def get_random_model():
    id_, _, sent, left, right, technique = get_random_technique()
    lst = ['' for _ in range(len(sent))]
    lst[left:right] = [technique for _ in range(left, right)]
    return jsonify(result={'id': id_, 'sent': sent, 'list': lst})
