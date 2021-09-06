import torch
import math
from torch import nn
import numpy as np
from transformers import BertConfig, BertModel
from utils.model_utils import Label_Fusion_Layer_for_Seq, Classifier_Layer, Label_Fusion_Layer_for_Classification


class BertForMultiSequenceClassification(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertForMultiSequenceClassification, self).__init__()
        self.label_num = args.first_label_num
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.dropout = nn.Dropout(args.dropout_rate)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.classifier = nn.Linear(
            self.encoder_config["hidden_size"], self.label_num)
        self.loss_fct = nn.BCELoss()

    def forward(self, data):
        out = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])

        # input_mask = data['input_mask'].unsqueeze(-1)
        # features = torch.mul(
        #     out[0], input_mask)
        # seq_features = torch.sum(features, dim=1) / \
        #     torch.sum(input_mask, dim=1)
        logits = self.classifier(out[1])
        # logits = self.classifier(seq_features)
        infers = torch.sigmoid(logits)
        output = (infers,)
        if 'first_types' in data:
            loss = self.loss_fct(infers, data['first_types'])
            output += (loss,)
        return output

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text, pooled_output = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[:2]
        encoded_text = self.dropout(encoded_text)
        pooled_output = self.dropout(pooled_output)
        return (encoded_text, pooled_output)


class BertForMultiSequenceClassificationWithLabel(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertForMultiSequenceClassificationWithLabel, self).__init__()
        self.label_num = args.first_label_num
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.hidden_size = self.encoder_config['hidden_size']
        self.dropout = nn.Dropout(args.dropout_rate)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        # self.classifier = nn.Linear(
        #     self.encoder_config["hidden_size"] * self.label_num, self.label_num)
        self.classifier = Classifier_Layer(self.label_num, self.hidden_size)
        self.label_fusing_layer = Label_Fusion_Layer_for_Seq(
            self.encoder_config["hidden_size"], self.label_num)
        self.label_type_emb_layer = nn.Embedding(self.label_num, 300)
        self.label_type_emb_layer.weight.data.copy_(torch.from_numpy(
            np.load(args.glove_label_emb_file, allow_pickle=True)))
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, data):
        out = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])
        label_embs = self.label_type_emb_layer(data['first_label_ids'])

        fused_feature = self.label_fusing_layer(out[1], label_embs)
        logits = self.classifier(fused_feature)
        output = (logits,)
        if 'first_types' in data:
            loss = self.loss_fct(logits, data['first_types'])
            output += (loss,)
        return output

    def get_text_embedding(self, token_ids, token_type_ids, input_mask, event_str_ids=None):
        encoded_text, pooled_output = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[:2]
        encoded_text = self.dropout(encoded_text)
        pooled_output = self.dropout(pooled_output)
        return (encoded_text, pooled_output)


class BertForMultiSequenceClassificationWithLabelSiamese(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertForMultiSequenceClassificationWithLabelSiamese, self).__init__()
        self.label_num = args.first_label_num
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.hidden_size = self.encoder_config['hidden_size']
        self.dropout = nn.Dropout(args.dropout_rate)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        # self.classifier = Classifier_Layer(self.label_num, self.hidden_size)
        self.classifier = nn.Linear(
            self.hidden_size, self.label_num)
        self.label_fusing_layer = Label_Fusion_Layer_for_Classification(
            self.hidden_size, self.label_num, self.hidden_size)
        self.loss_fct = nn.BCELoss()

    def forward(self, data, label_token_ids, label_token_type_ids, label_input_mask):
        out = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])[1]
        # batch_size = out[0].shape[0]
        # encoded_label = self.get_text_embedding(
        #     data['label_token_ids'], data['label_token_type_ids'], data['label_input_mask'])[0]

        label_embs = self.get_text_embedding(
            label_token_ids, label_token_type_ids, label_input_mask)[1]

        # label_input_mask_extended = data['label_input_mask'].unsqueeze(
        #     -1)
        # # [class_num, hidden_dim]
        # label_embs = torch.sum(encoded_label * label_input_mask_extended,
        #                        1) / torch.sum(label_input_mask_extended, dim=1)
        # label_embs = label_embs.reshape(
        #     self.label_num, self.hidden_size).unsqueeze(0).repeat(batch_size, 1, 1)
        # label_embs = self.label_type_emb_layer(data['first_label_ids'])

        fused_feature = self.label_fusing_layer(
            out[0], label_embs, input_mask=data['input_mask'])[0]
        logits = self.classifier(fused_feature)
        output = (logits,)
        if 'first_types' in data:
            loss = self.loss_fct(logits, data['first_types'])
            output += (loss,)
        return output

    def get_text_embedding(self, token_ids, token_type_ids, input_mask, event_str_ids=None):
        encoded_text, pooled_output = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[:2]
        encoded_text = self.dropout(encoded_text)
        pooled_output = self.dropout(pooled_output)
        return (encoded_text, pooled_output)
