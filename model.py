import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from torch import nn
from torch.nn.functional import relu, sigmoid

from data_load import NUM_TASK, VOCAB, MASKING, HIER
from hp import SIGMOID_ACTIVATION


class BertMultiTaskLearning(PreTrainedBertModel):
    def __init__(self, config, **dct):

        super().__init__(config)

        if dct is None:
            dct = {
                'NUM_TASK': NUM_TASK, 'MASKING': MASKING,
                'SIGMOID_ACTIVATION': SIGMOID_ACTIVATION, 'HIER': HIER, 'VOCAB': VOCAB,
            }

        self.NUM_TASK = dct['NUM_TASK']
        self.MASKING = dct['MASKING']
        self.SIGMOID_ACTIVATION = dct['SIGMOID_ACTIVATION']
        self.HIER = dct['HIER']

        self.VOCAB = [
            ("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans",
             "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation",
             "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon",
             "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion",
             "Appeal_to_Authority", "Black-and-White_Fallacy",
             "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism")
        ]

        if self.NUM_TASK == 2:  # sentence classification
            self.VOCAB.append(("Non-prop", "Prop"))

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, len(self.VOCAB[i]))
                                         for i in range(self.NUM_TASK)])
        self.apply(self.init_bert_weights)
        self.masking_gate = nn.Linear(2, 1)

        if self.NUM_TASK == 2:
            self.merge_classifier_1 = nn.Linear(
                len(self.VOCAB[0]) + len(self.VOCAB[1]), len(self.VOCAB[0])
            )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        if self.NUM_TASK == 1:
            logits = [self.classifier[i](sequence_output) for i in range(self.NUM_TASK)]
        elif self.NUM_TASK == 2 and self.MASKING:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)

            if self.SIGMOID_ACTIVATION:
                gate = sigmoid(self.masking_gate(sen_level))
            else:
                gate = relu(self.masking_gate(sen_level))

            dup_gate = gate.unsqueeze(1).repeat(1, token_level.size()[1], token_level.size()[2])
            wei_token_level = torch.mul(dup_gate, token_level)

            logits = [wei_token_level, sen_level]
        elif self.NUM_TASK == 2 and self.HIER:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)
            dup_sen_level = sen_level.repeat(1, token_level.size()[1])
            dup_sen_level = dup_sen_level.view(sen_level.size()[0], -1, sen_level.size()[-1])
            logits = [
                self.merge_classifier_1(torch.cat((token_level, dup_sen_level), 2)),
                self.classifier[1](pooled_output)
            ]
        elif self.NUM_TASK == 2:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)
            logits = [token_level, sen_level]
        else:
            raise ValueError("Incorrect combination of input arguments")
        y_hats = [logits[i].argmax(-1) for i in range(self.NUM_TASK)]

        return logits, y_hats
