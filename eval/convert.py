import pathlib
import sys
from collections import defaultdict
from operator import itemgetter

BASE_DIR = pathlib.Path.cwd()
DIRECTORY_TRAIN = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'train')
DIRECTORY_DEV = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'dev')
DIRECTORY_TEST = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'test')
DIRECTORY_MARKUP = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'markup')
DIRECTORY_PREDICT = BASE_DIR.joinpath('data_ru', 'protechn_corpus_eval', 'predict')
ARTICLE = 7
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


def get_list(id_, directory=DIRECTORY_TRAIN):
    """
    Функция, возвращающая список [set(), set(), ..., {Flag-Waving, Bandwagon}, ..., set(), set()].
    """

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


def get_id_to_x():
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


def read_data(directory):
    ids = []
    texts = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt', '')
        ids.append(id)
        texts.append(f.read_text(encoding='utf-8'))
    return ids, texts


def clean_text(articles, ids):
    texts = []
    for article, id in zip(articles, ids):
        sentences = article.split('\n')
        start = 0
        end = -1
        res = []
        for sentence in sentences:
            start = end + 1
            end = start + len(sentence)  # length of sequence
            if sentence != "":  # if not empty line
                res.append([id, sentence, start, end])
        texts.append(res)
    return texts


def check_overlap(line_1, line_2):
    if line_1[2] > line_2[3] or line_1[3] < line_2[2]:
        return False
    return True


def remove_duplicates(res):
    sorted_res = sorted(res, key=itemgetter(0, 1, 2, 3))
    ans = []
    skip = 0
    for i, line_1 in enumerate(sorted_res):
        assert line_1 == sorted_res[i]
        for j, line_2 in enumerate(sorted_res[i + 1:]):
            skip = 0
            if line_1[0] != line_2[0]:
                break
            elif line_1[1] != line_2[1]:
                continue

            if check_overlap(line_1, line_2):
                if line_1[2] != line_2[2] or line_1[3] != line_2[3]:
                    sorted_res[i + j + 1][2] = min(line_1[2], line_2[2])
                    sorted_res[i + j + 1][3] = max(line_1[3], line_2[3])
                skip = 1
                break
        if skip == 0:
            ans.append(line_1)
    return ans


def convert(ind, flat_texts, filename):
    """
    1173236160
    Мало O Non-prop O Prop
    того, O Non-prop O Prop
    что O Non-prop O Prop
    «Аркан» O Non-prop O Prop
    приобрел O Non-prop O Prop
    «Ведомости» O Non-prop O Prop
    у O Non-prop O Prop
    кипрского O Non-prop O Prop
    офшора O Non-prop O Prop
    за O Non-prop O Prop
    бóльшую O Non-prop Loaded_Language Prop
    сумму, O Non-prop O Prop
    чем O Non-prop O Prop
    они O Non-prop O Prop
    <...>

    1173236160
    <...>
    """

    id_to_text, *_ = get_id_to_x()

    with open(filename, 'r', encoding='utf-8') as f1:
        output = []
        for line in f1:
            if len(line.split()) == 1:  # if line is id
                id_ = line.strip()
                continue
            elif line != '\n':  # In the same sentence
                tmp = [id_] + line.strip().split()  # add id to line
                if len(tmp) == 6:  # num_task 2
                    tmp += [tmp[-2]]
                else:
                    tmp += [tmp[-(1 + ind)]]
                output.append(tmp + [len(tmp[1])])  # add word length to line
            else:
                output.append('\n')

    res = []
    aid = output[0][0]
    sub_list = [sentence for sentence in flat_texts if sentence[0] == aid]
    sub_dic = {sentence: (start, end) for _, sentence, start, end in sub_list}

    start = 0
    end = -1
    sentence = ""
    cur = 0
    on = 0

    tmp_ans = []
    cur_tag = 'O'
    prop_or_not_dict = {}
    for line in output:  # ['36081082999', 'вора', 'O', 'Non-prop', 'O', 'Non-prop', 'O', 4]
        if line != '\n':
            aid = line[0]
            if int(aid) not in prop_or_not_dict:
                prop_or_not_dict[int(aid)] = [False for _ in range(len(id_to_text[int(aid)]))]

            sentence += line[1] + " "
            prop_or_not_prop = line[-3] != 'Non-prop'
            if line[-2] != 'O' and line[-2] != '<PAD>':
                if on == 0:
                    on = 1
                    cur_tag = line[-2]
                    start = cur
                    end = cur + line[-1]
                elif line[-2] == cur_tag:
                    end = cur + line[-1]
                else:
                    tmp_ans.append([aid, cur_tag, start, end])
                    cur_tag = line[-2]
                    start = cur
                    end = cur + line[-1]
            else:
                if on:
                    tmp_ans.append([aid, cur_tag, start, end])
                    on = 0
            cur += line[-1] + 1

        else:
            if on:
                tmp_ans.append([aid, cur_tag, start, end])
                on = 0

            cur = 0
            sub_list = [sentence for sentence in flat_texts if sentence[0] == aid]
            sub_dic = {sentence: (start, end) for _, sentence, start, end in sub_list}

            if len(tmp_ans) and sentence[:-1] != "":
                s, e = sub_dic.get(sentence[:-1])
                if prop_or_not_prop:
                    prop_or_not_dict[int(aid)][s:e] = [True for _ in range(s, e)]

            if len(tmp_ans) and sentence[:-1] != "":
                for ans in tmp_ans:
                    ans[2] += s
                    ans[3] += s
                    res.append(ans)
            sentence = ""
            prop_or_not_prop = False
            tmp_ans = []
    return res, prop_or_not_dict


if __name__ == "__main__":

    directory = pathlib.Path('./data/protechn_corpus_eval/test')
    ids, texts = read_data(directory)

    t_texts = clean_text(texts, ids)
    flat_texts = [sentence for article in t_texts for sentence in article]

    id_ind = [sentence[0] for sentence in flat_texts]

    fi = []
    if sys.argv[2] == 'bert':
        fi = convert(0, flat_texts, sys.argv[1])
    elif sys.argv[2] == 'bert-joint' or sys.argv[2] == 'bert-granu' or sys.argv[2] == 'mgn':
        fi = convert(1, flat_texts, sys.argv[1])

    print(f'fi: {fi}')
    res = remove_duplicates(fi)
    print(f'res: {res}')

    with open("./eval/official_prediction.txt", 'w', encoding='utf-8') as f3:
        for i in res:
            f3.write("\t".join([i[0], i[1], str(i[2]), str(i[3])]) + "\n")
