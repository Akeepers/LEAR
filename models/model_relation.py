import torch
from torch import nn
from transformers import *
import torch.nn.functional as F
from utils.losses import *
from utils.model_utils import Label_Fusion_Layer_for_Token, Label_Attn_Fusion_Layer_for_token, Classifier_Layer


def loss_v2(gold, infer, active_pos, loss_mask=None):
    if loss_mask is not None:
        loss_mask = loss_mask.view(-1, 1, 1)
        infer = infer * loss_mask
        gold = gold * loss_mask
    label_num = infer.shape[-1]
    # active_pos = padding_mask.contiguous().view(-1) == 1
    masked_infer = infer.contiguous().view(-1, label_num)[active_pos]
    masked_gold = gold.contiguous().view(-1, label_num)[active_pos]
    loss_ = F.binary_cross_entropy(
        masked_infer, masked_gold, reduction='none')
    loss_ = torch.sum(loss_, 1)
    loss = torch.mean(loss_)
    return loss


class Loss_func(nn.Module):
    def __init__(self, strategy='v2'):
        super(Loss_func, self).__init__()
        # if strategy == 'v2':
        #     self.loss_func = loss_v2
        # elif strategy == 'v1':
        #     self.loss_func == loss_v1

    def forward(self, data, infer_sub_heads, infer_sub_tails, infer_obj_heads, infer_obj_tails):
        active_pos = data['input_mask'].contiguous().view(-1) == 1
        sub_heads_loss = self.loss_func(
            data['first_starts'], infer_sub_heads, active_pos)
        sub_tails_loss = self.loss_func(
            data['first_ends'], infer_sub_tails, active_pos)
        obj_heads_loss = self.loss_func(
            data['second_starts'], infer_obj_heads, active_pos)
        obj_tails_loss = self.loss_func(
            data['second_ends'], infer_obj_tails, active_pos)
        return sub_heads_loss + sub_tails_loss + obj_heads_loss + obj_tails_loss

    def loss_func(self, gold, infer, active_pos, loss_mask=None):
        if loss_mask is not None:
            loss_mask = loss_mask.view(-1, 1, 1)
            infer = infer * loss_mask
            gold = gold * loss_mask
        label_num = infer.shape[-1]
        masked_infer = infer.contiguous().view(-1, label_num)[active_pos]
        masked_gold = gold.contiguous().view(-1, label_num)[active_pos]
        loss_ = F.binary_cross_entropy(
            masked_infer, masked_gold, reduction='none')
        loss_ = torch.sum(loss_, 1)
        loss = torch.mean(loss_)
        return loss


class BertRelationSpan(nn.Module):
    def __init__(self, args):
        super(BertRelationSpan, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.dropout = nn.Dropout(0.1)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.first_head_classifier = nn.Linear(
            self.encoder_config["hidden_size"], args.first_label_num)
        self.first_tail_classifier = nn.Linear(
            self.encoder_config["hidden_size"], args.first_label_num)
        self.second_head_classifier = nn.Linear(
            self.encoder_config["hidden_size"], args.second_label_num)
        self.second_tail_classifier = nn.Linear(
            self.encoder_config["hidden_size"], args.second_label_num)
        self.first_label_feature_fc = nn.Sequential(
            nn.Linear(self.encoder_config["hidden_size"],
                      self.encoder_config["hidden_size"]),
            nn.Tanh()
        )
        self.loss_func = Loss_func()

    def forward(self, data):
        encoded_text = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])
        infer_sub_heads, infer_sub_tails = self.first_label_extractor(
            encoded_text)

        sub_head_mapping = data['first_start'].unsqueeze(1)
        sub_tail_mapping = data['first_end'].unsqueeze(1)

        infer_obj_heads, infer_obj_tails = self.second_label_extractor(data, sub_head_mapping, sub_tail_mapping,
                                                                       encoded_text)
        loss = self.loss_func(data, infer_sub_heads, infer_sub_tails,
                              infer_obj_heads, infer_obj_tails)
        return (infer_sub_heads, infer_sub_tails, infer_obj_heads, infer_obj_tails, loss)

    def second_label_extractor(self, data, sub_head_mapping, sub_tail_mapping, encoded_text):
        first_head = torch.matmul(sub_head_mapping, encoded_text)
        second_tail = torch.matmul(sub_tail_mapping, encoded_text)

        first_label_feature = (first_head + second_tail) / 2

        first_label_feature = self.first_label_feature_fc(
            first_label_feature)
        encoded_text = encoded_text + first_label_feature

        infer_obj_heads = self.second_head_classifier(encoded_text)
        infer_obj_heads = torch.sigmoid(infer_obj_heads)

        infer_obj_tails = self.second_tail_classifier(encoded_text)
        infer_obj_tails = torch.sigmoid(infer_obj_tails)
        return infer_obj_heads, infer_obj_tails

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        encoded_text = self.dropout(encoded_text)
        return encoded_text

    def first_label_extractor(self, encoded_text):
        logits_head = self.first_head_classifier(encoded_text)
        infer_first_head = torch.sigmoid(logits_head)

        logits_tail = self.first_tail_classifier(encoded_text)
        infer_first_tail = torch.sigmoid(logits_tail)
        return infer_first_head, infer_first_tail


class BertRelationSpanLabelSiamese(nn.Module):
    def __init__(self, args):
        super(BertRelationSpanLabelSiamese, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.dropout = nn.Dropout(0.1)
        self.label_num = args.second_label_num
        self.hidden_size = self.encoder_config['hidden_size']
        self.label_emb_dim = args.label_emb_size
        self.use_auxiliary_task = args.use_auxiliary_task
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.first_head_classifier = nn.Linear(
            self.encoder_config["hidden_size"] * self.label_num, args.first_label_num)
        self.first_tail_classifier = nn.Linear(
            self.encoder_config["hidden_size"] * self.label_num, args.first_label_num)
        self.second_head_classifier = Classifier_Layer(
            self.label_num, self.hidden_size)
        self.second_tail_classifier = Classifier_Layer(
            self.label_num, self.hidden_size)
        self.first_label_feature_fc = nn.Sequential(
            nn.Linear(self.encoder_config["hidden_size"],
                      self.encoder_config["hidden_size"]),
            nn.Tanh()
        )
        self.loss_func = Loss_func()
        if self.use_auxiliary_task:
            pass
        else:
            self.label_fusing_layer = Label_Fusion_Layer_for_Token(
                self.encoder_config["hidden_size"], self.label_num, self.label_emb_dim)

    def forward(self, data, label_token_ids, label_token_type_ids, label_input_mask):
        encoded_text = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])
        batch_size, seq_len = encoded_text.shape[:2]

        # encoded_label = self.get_text_embedding(
        #     data['label_token_ids'], data['label_token_type_ids'], data['label_input_mask'])  # [class_num, seq_len, hidden_dim]

        # [class_num, hidden_dim]
        # label_embs = torch.sum(
        #     torch.mul(encoded_label, data['label_input_mask'].unsqueeze(-1)), 1)
        # label_embs = label_embs.unsqueeze(0).repeat(batch_size, 1, 1)

        # fused_results = self.label_fusing_layer(
        #     encoded_text, label_embs, return_label_embs=self.use_auxiliary_task)  # [bs , seq_len, label_num, hidden_dim]

        label_embs = self.get_text_embedding(
            label_token_ids, label_token_type_ids, label_input_mask)

        fused_results = self.label_fusing_layer(
            encoded_text, label_embs, data['input_mask'], label_input_mask=label_input_mask, use_attn=True)

        infer_sub_heads, infer_sub_tails = self.first_label_extractor(
            fused_results[0].contiguous().view(batch_size, seq_len, -1))

        sub_head_mapping = data['first_start'].unsqueeze(1)
        sub_tail_mapping = data['first_end'].unsqueeze(1)

        infer_obj_heads, infer_obj_tails = self.second_label_extractor(data, sub_head_mapping, sub_tail_mapping,
                                                                       fused_results[0])
        loss = self.loss_func(data, infer_sub_heads, infer_sub_tails,
                              infer_obj_heads, infer_obj_tails)
        return (infer_sub_heads, infer_sub_tails, infer_obj_heads, infer_obj_tails, loss)

    def second_label_extractor(self, data, sub_head_mapping, sub_tail_mapping, encoded_text):
        batch_size, seq_len = encoded_text.shape[:2]
        encoded_text_ = encoded_text.contiguous().view(batch_size, seq_len, -1)

        # [bs, 1, seq_len] , [bs, seq_len*class_num, hidden_dim] -> [bs, 1, class_num, hidden_dim]

        first_head = torch.matmul(sub_head_mapping, encoded_text_).contiguous().view(
            batch_size, 1, self.label_num, self.hidden_size)
        second_tail = torch.matmul(sub_tail_mapping, encoded_text_).contiguous().view(
            batch_size, 1, self.label_num, self.hidden_size)

        first_label_feature = (first_head + second_tail) / 2

        first_label_feature = self.first_label_feature_fc(
            first_label_feature)
        encoded_text = encoded_text + first_label_feature

        encoded_text = encoded_text.contiguous().view(-1, self.label_num, self.hidden_size)

        infer_obj_heads = self.second_head_classifier(
            encoded_text).contiguous().view(batch_size, seq_len, self.label_num)
        infer_obj_heads = torch.sigmoid(infer_obj_heads)

        infer_obj_tails = self.second_tail_classifier(
            encoded_text).contiguous().view(batch_size, seq_len, self.label_num)
        infer_obj_tails = torch.sigmoid(infer_obj_tails)
        return infer_obj_heads, infer_obj_tails

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        encoded_text = self.dropout(encoded_text)
        return encoded_text

    def first_label_extractor(self, encoded_text):
        logits_head = self.first_head_classifier(encoded_text)
        infer_first_head = torch.sigmoid(logits_head)

        logits_tail = self.first_tail_classifier(encoded_text)
        infer_first_tail = torch.sigmoid(logits_tail)
        return infer_first_head, infer_first_tail
