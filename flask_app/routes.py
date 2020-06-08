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


@app.route('/info', methods=['GET'])
def info():
    techniques = [
        {
            'id': 1,
            'technique': """Ad nauseam
Repetition""", 'include': '✔️',
            'text': """И он работает достаточно эффективно, <b>растёт</b> и экспорт этих наших услуг и товаров — <b>растёт</b>. (#36163344507)
И я верю, <b>люди очень искренние</b>. <b>Очень искренние люди</b> у нас. (#91678575004)""",
        },
        {
            'id': 2,
            'technique': """Appeal to authority
Testimonial""", 'include': '✔️',
            'text': """<b>Всем очевидно</b>, что в нынешней ситуации борьба профсоюза за улучшение условий труда выгодна всем и приумножает общее благо. (#12402123807)
В 600 тысяч рублей <b>журналисты оценили</b> один рекламный контракт актера. (#83366723989)"""
        },
        {
            'id': 3,
            'technique': """Appeal to fear
Appeal to prejudice""",
            'include': '✔️',
            'text': """Читаешь и понимаешь: да, нам нужен алармизм и вопящие от негодования профсоюзы, 
потому что <b>иначе государство и не пошевелится</b>. (#12402123807)
От плановых операций в больницах надо воздержаться — <b>очень высокая вероятность заражения</b>. (#86789309327)""",
        },
        {
            'id': 4,
            'technique': 'Bandwagon',
            'include': '✔️',
            'text': """<b>Во всем мире так поступают</b>. (#3490019195)
<b>Или мы будем исключением?</b> (#49985942642)""",
        },
        {
            'id': 5,
            'technique': """Beautiful people
Guilt by association / Reductio ad Hitlerum
Transfer
Euphoria
Operant conditioning
Classical conditioning""",
            'include': '✔️',
            'text': """Извините, <b>в некоторых странах жёлтые звёзды евреям пришивали</b>. (#53424346461)""",
        },
        {
            'id': 6,
            'technique': """Black-and-white fallacy
Dictat(orship)""",
            'include': '✔️',
            'text': """<b>Невозможно всех удержать. Или не надо столько готовить</b>. (#33748247649)
<b>А ничего авторитарнее армии не существует</b>. (#98031283058)""",
        },
        {
            'id': 7,
            'technique': """Common man
Cult of personality""",
            'include': '❌',
            'text': """Стратегии по созданию нужного образа у человека, не является техникой в используемом нами смысле.""",
        },
        {
            'id': 8,
            'technique': """Agenda setting
Big lie
Demoralization
Divide and rule
Firehose of falsehood
Framing
Managing the news
Third party technique""",
            'include': '❌',
            'text': """Все эти техники являются стратегиями по успешному распространению пропаганды.""",
        },
        {
            'id': 9,
            'technique': """Door-in-the-face
Foot-in-the-door
Latitudes of acceptance""",
            'include': '❌',
            'text': """Стратегии по применению уступок с целью продать свой товар по нужной цене (обобщается и на идеи, но встречается редко)""",
        },
        {
            'id': 10,
            'technique': """Exaggeration
Minimisation""",
            'include': '✔️',
            'text': """И тем более никто из них <b>по 20 раз в день не занимается самовосхвалением, расписывая в соцсетях каждый эпизод своей «гуманитарной» деятельности</b>. (#41105096806)
Они <b>сто раз пожалели</b>, что продали. (#49985942642)
«Разумеется, общество у нас глубоко патриархальное, и поэтому всегда женщина в нем все должна, а <b>мужчина ничего не должен</b>. (#83366723989)""",
        },
        {
            'id': 11,
            'technique': """Fear, uncertainty, and doubt""",
            'include': '✔️',
            'text': """Вы же расследователь <b>якобы</b>. (#7838448925)
В начале апреля представители «Альянса врачей» приехали в ЦГКБ Реутова, чтобы <b>«помочь»</b> с защитными средствами. (#52498183368)""",
        },
        {
            'id': 12,
            'technique': """Flag-waving""",
            'include': '✔️',
            'text': """<b>Мы реально обогнали всех</b>, мы больше всех делаем в мире атомных блоков для электростанций. (#95967168572)
<b>Это не наш путь</b>. (#3490019195)""",
        },
        {
            'id': 13,
            'technique': """Gaslighting
Gish gallop
Information overload""",
            'include': '❌',
            'text': """Воздействие на ограничения восприятия и внимания человека с целью наиболее эффективного распространения пропаганды.""",
        },
        {
            'id': 14,
            'technique': """Cognitive dissonance""",
            'include': '❌',
            'text': """Психологическая стратегия вызвать у слушающего диссонанс, чтобы он принял ту или иную точку зрения.""",
        },
        {
            'id': 15,
            'technique': """Glittering generalities
Pensée unique
Thought-terminating clichés
Rationalization""",
            'include': '✔️',
            'text': """<b>Ну чего же здесь хорошего?</b> (#2382492813)
<b>Ну это естественно</b>. (#49985942642)
«<b>По сути</b>, идет война» (#40120334507)
<b>Бог им судья</b>. (#40120334507)
К сожалению, <b>такая реальность</b>. (#86789309327)
""",
        },
        {
            'id': 16,
            'technique': """False accusations
Half-truth
Lying and deception
Smears
Disinformation""",
            'include': '❌',
            'text': """Для данных техник необходим внешний контекст, а именно знание о том, что есть истина, а что — ложь.""",
        },
        {
            'id': 17,
            'technique': """Intentional vagueness
Obfuscation, intentional vagueness, confusion""",
            'include': '✔️',
            'text': """Но на первом этапе это не просто взять и выдрать деньги с человека, <...>, чтобы <b>государство не подставляло его от раза к разу под какую-то статью</b>. (#1241238761)""",
        },
        {
            'id': 18,
            'technique': """Ad hominem
Dysphemism
Labeling
Name-calling
Stereotyping""",
            'include': '✔️',
            'text': """Мы же понимаем, что это <b>чушь</b>. (#59051731723)
Вот поэтому-то несчастный директор и изображает из себя <b>зоологического русофоба</b>. (#69294925216)
Только за жизни людей отвечают не вот эти <b>горластые придурки</b>, которые прыгают по разного рода ютубам, а руководство страны. (#8359563559)""",
        },
        {
            'id': 19,
            'technique': """Demonizing the enemy
Loaded language""",
            'include': '✔️',
            'text': """<b>Несут всякую фигню</b> в Европарламенте по поводу одинаковой ответственности Гитлера и Сталина. (#59051731723)
Навальный вчера <b>яро</b> защищал Быкова. (#5326402550)""",
        },
        {
            'id': 20,
            'technique': """Love bombing
Milieu control""",
            'include': '❌',
            'text': """Стратегии по привлечению людей в сообщество и удержанию их в нём.""",
        },
        {
            'id': 21,
            'technique': """Non sequitur""",
            'include': '❌',
            'text': """Используется в большом количестве техник, основополагающая, но слишком общая техника.""",
        },
        {
            'id': 22,
            'technique': """(Causal) Oversimplification
Scapegoating""",
            'include': '✔️',
            'text': """<b>У нас просто меньше людей, которые в детородном возрасте находятся</b>. (#68462833391)
Он отказался, <b>это было его решение</b>. (#40120334507)
<b>Это пускай они скажут спасибо Михаилу Николаевичу</b>. (#97506920380)""",
        },
        {
            'id': 23,
            'technique': """Cherry picking
Quotes out of context""",
            'include': '❌',
            'text': """Рассмотрение каких-либо фраз или положений вне контекста.""",
        },
        {
            'id': 24,
            'technique': "Red herring",
            'include': '✔️',
            'text': """«Ситуацию, которая сейчас разворачивается вокруг реутовской больницы, считаю заранее спланированной — <b>в тот момент, когда мы круглосуточно боремся за жизни пациентов в это непростое время</b>», — сказал он. (#52498183368)""",
        },
        {
            'id': 25,
            'technique': "Slogans",
            'include': '✔️',
            'text': """Ну, 9 Мая типа на BMW или на Mercedes сзади: «<b>Можем повторить</b>», «<b>На Берлин</b>»… (#59051731723)""",
        },
        {
            'id': 26,
            'technique': "Straw man",
            'include': '✔️',
            'text': """<b>Главное обвинение RT состоит в том, что у него слишком много просмотров</b>. (#98031283058)""",
        },
        {
            'id': 27,
            'technique': "Unstated assumption",
            'include': '❌',
            'text': """Техника заключается в неявном подразумевании, а не в непосредственном применении какой-либо техники, поэтому автоматически её детектировать не представляется возможным.""",
        },
        {
            'id': 28,
            'technique': """Euphemism
Virtue words""",
            'include': '❌',
            'text': """Формальная неотличимость нейтральных контекстов и контекстов с применением техники.""",
        },
        {
            'id': 29,
            'technique': "Whataboutism",
            'include': '✔️',
            'text': """<b>А они теряют рабочие места</b>. (#31808762171)
<b>Но если Вы вспомните, что было в 90-х годах</b>, Вы же взрослый человек, ну бардак это был, а не парламент. (#73936725916)
""",
        },
    ]

    return render_template('info.html', techniques=techniques)


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
