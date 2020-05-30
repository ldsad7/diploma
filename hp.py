from pathlib import Path

TRAINING = "training"
CHECKDIR = "checkpoints"
RESULTDIR = "results"

# either of the four variants:
BERT = False
JOINT = False
GRANU = False
MGN = True

assert BERT or JOINT or GRANU or MGN
assert not (BERT and JOINT) and not (BERT and GRANU) and not (BERT and MGN) \
       and not (JOINT and GRANU) and not (JOINT and MGN) and not (GRANU and MGN)

# either of the two variants
SIGMOID_ACTIVATION = True
RELU_ACTIVATION = False
assert not (SIGMOID_ACTIVATION and RELU_ACTIVATION) and (SIGMOID_ACTIVATION or RELU_ACTIVATION)

TRAINSET = Path('./data_ru/protechn_corpus_eval/train')
VALIDSET = Path('./data_ru/protechn_corpus_eval/dev')
TESTSET = Path('./data_ru/protechn_corpus_eval/test')

BATCH_SIZE = 32
LR = 1e-5
ALPHA = 0.75
N_EPOCHS = 100
PATIENCE = 15
INPUT_SIZE = 768
SEQ_LEN = 212
POS_WEIGHT = 926 / 3532

BERT_PATH = 'BERT_ru'
JOINT_BERT_PATH = 'BERT_JOINT_model_ru'
GRANU_BERT_PATH = 'BERT_GRAN_model_ru'
MGN_SIGM_BERT_PATH = 'BERT_MULTIGRAN_model_sigmoid_ru'
MGN_RELU_BERT_PATH = 'BERT_MULTIGRAN_model_relu_ru'
