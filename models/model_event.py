import torch
import math
from torch import nn
import numpy as np
from torch.autograd import Variable
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, AutoModelForTokenClassification, AutoModel, AutoConfig
# from transformers.modeling_bert import BertSelfAttention
from utils.model_utils import GGNN_1, Classifier_Layer, ACT_basic, SelfAttention, _gen_timing_signal, Label_Fusion_Layer_for_Token, Label_Attn_Fusion_Layer_for_token
from utils.crf import CRF


class OnePassIE(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(OnePassIE, self).__init__()
        self.add_first_feature_fc = args.add_first_feature_fc
        self.task_type = args.task_type
        self.add_act = args.add_act
        self.add_gate = args.add_gate
        self.add_cls = args.add_cls
        self.add_attention = args.add_attention
        self.add_transformer = args.add_transformer
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.hidden_size = self.encoder_config['hidden_size']
        self.add_event_type = args.add_event_type
        self.use_event_str = args.use_event_str
        if self.task_type == 'event':
            self.first_label_extra_info_embedding = nn.Embedding(
                2, self.hidden_size)
        elif self.task_type == "relation":
            self.first_label_extra_info_embedding = nn.Embedding(
                args.max_seq_length * 2-1, self.hidden_size)
        if self.add_act:
            # temp (proj func)
            self.embedding_proj = nn.Linear(
                self.encoder_config["hidden_size"], self.encoder_config["hidden_size"], bias=False)
            self.input_dropout = nn.Dropout(input_dropout)
            self.timing_signal = _gen_timing_signal(
                args.max_seq_length, self.encoder_config["hidden_size"])
            self.position_signal = _gen_timing_signal(
                self.encoder_config['num_hidden_layers'], self.encoder_config["hidden_size"])
            self.act_fc = ACT_basic(self.encoder_config["hidden_size"])
        if self.add_gate:
            self.gate = nn.Sequential(
                nn.Linear(
                    self.hidden_size+self.encoder_config['hidden_size'], self.encoder_config['hidden_size']),
                nn.Sigmoid()
            )
            self.gate_cls = nn.Sequential(
                nn.Linear(
                    self.hidden_size+self.encoder_config['hidden_size'], self.encoder_config['hidden_size']),
                nn.Sigmoid()
            )
        self.dropout = nn.Dropout(args.dropout_rate)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.first_head_classifier = nn.Linear(
            self.encoder_config["hidden_size"] * 2, args.first_label_num)
        self.first_tail_classifier = nn.Linear(
            self.encoder_config["hidden_size"] * 2, args.first_label_num)
        self.first_type_classifier = nn.Linear(
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
        # self.multi_head_attn_layer = SelfAttention(
        #     self.encoder_config["hidden_size"], self.encoder_config["num_attention_heads"], self.encoder_config['attention_probs_dropout_prob'])
        self.multi_head_attn_layer = nn.MultiheadAttention(
            self.encoder_config["hidden_size"], self.encoder_config["num_attention_heads"], self.encoder_config['attention_probs_dropout_prob'])

        self.event_multi_head_attn_layer = SelfAttention(
            self.encoder_config["hidden_size"], self.encoder_config["num_attention_heads"], self.encoder_config['attention_probs_dropout_prob'])
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            self.encoder_config["hidden_size"], self.encoder_config["num_attention_heads"])
        self.glove_dim = 300
        self.event_type_embedding = nn.Embedding(
            args.first_label_num, self.glove_dim)
        glove_event_file = '/home/yangpan/workspace/onepass_ie/data/ace05/splited/glove_event_type_uncased_whole_300d.npy'
        emb = torch.from_numpy(
            np.load(glove_event_file, allow_pickle=True))
        emb = emb.to(args.device)
        self.event_type_embedding.weight.data.copy_(emb)
        self.word_dim = self.glove_dim
        self.event_trans = nn.Sequential(
            nn.Linear(self.word_dim,
                      self.encoder_config["hidden_size"]),
            nn.ReLU()
        )

        # event emb attention:
        self.event_add_attention = AddAttention(self.hidden_size)

    def forward(self, data):
        out = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'], data['event_str_ids'])
        encoded_text = out[0]
        if self.add_event_type:
            first_output = self.first_label_extractor(
                encoded_text, out[1], data)
        else:
            first_output = self.first_label_extractor(encoded_text)
        infer_first_starts, infer_first_ends = first_output[0], first_output[1]

        first_start_mapping = data['first_start'].unsqueeze(1)
        first_end_mapping = data['first_end'].unsqueeze(1)

        infer_second_starts, infer_second_ends = self.second_label_extractor(data, first_start_mapping, first_end_mapping,
                                                                             encoded_text)
        output = (infer_first_starts, infer_first_ends,
                  infer_second_starts, infer_second_ends)
        if self.add_event_type:
            infer_first_label_type, first_label_type_loss = first_output[2], first_output[3]
            # if self.use_event_str:
            #     infer_first_label_type, first_label_type_loss = self.first_label_type_classifier(
            #         data, out[1])
            # else:
            #     infer_first_label_type, first_label_type_loss = self.first_label_type_classifier(
            #         data, out[2])
            output += (infer_first_label_type, first_label_type_loss)
        return output

    def second_label_extractor(self, data, first_start_mapping, first_end_mapping, encoded_text):
        first_head = torch.matmul(first_start_mapping, encoded_text)
        second_tail = torch.matmul(first_end_mapping, encoded_text)

        first_label_feature = (first_head + second_tail) / 2
        if self.add_first_feature_fc and self.task_type == 'relation':
            first_label_feature = self._mal(self.first_label_feature_fc(
                first_label_feature), data['second_label_mask'])
        first_label_feature = self._mal(
            first_label_feature, data['second_label_mask'])
        if self.task_type == 'event':
            first_label_extra_feature = self._mal(self.first_label_extra_info_embedding(
                data['text_type_ids']), data['second_label_mask'])
        elif self.task_type == 'relation':
            first_label_extra_feature = self._mal(self.first_label_extra_info_embedding(
                data['relative_pos_ids']), data['second_label_mask'])

        if self.add_first_feature_fc and self.task_type == 'event':
            first_label_extra_feature = self.first_label_feature_fc(
                first_label_extra_feature + first_label_feature)
        else:
            first_label_extra_feature = first_label_extra_feature + first_label_feature
        if self.add_gate:
            # first_label_extra_feature = first_label_extra_feature + first_label_feature
            # print(first_label_extra_feature.shape)
            concated = torch.cat((encoded_text, first_label_extra_feature), -1)
            f_gate = self.gate(concated)
            # print(f_gate.shape)
            # print(f_gate)
            # print(f_gate.shape)
            encoded_text = torch.add(
                torch.mul(first_label_extra_feature, f_gate), encoded_text)
        else:
            # if self.task_type == "event":
            #     first_label_extra_feature = self.first_label_feature_fc(
            #         first_label_extra_feature + first_label_feature)
            #     encoded_text = encoded_text + first_label_extra_feature
            # else:
            encoded_text = encoded_text + first_label_extra_feature

        if self.add_act:
            encoded_text = self.act_encoding(
                encoded_text, data['extra_mask'])[0]
        elif self.task_type == "event" and self.add_attention:
            encoded_text = encoded_text.permute(1, 0, 2)
            encoded_text = self.multi_head_attn_layer(
                query=encoded_text, key=encoded_text, value=encoded_text, key_padding_mask=data['extra_mask'].bool())[0]
            encoded_text = encoded_text.permute(1, 0, 2)
            # attn_mask = self.get_extended_attention_mask(
            #     data['input_mask'], data['token_ids'].size(), data['token_ids'].device)
            # encoded_text = self.multi_head_attn_layer(
            #     hidden_states=encoded_text, attention_mask=attn_mask)[0]
        elif self.task_type == "relation" and self.add_transformer:
            # attn_mask = self.get_extended_attention_mask(
            #     data['input_mask'], data['token_ids'].size(), data['token_ids'].device)
            # encoded_text = self.multi_head_attn_layer(
            #     hidden_states=encoded_text, attention_mask=attn_mask)[0]
            encoded_text = encoded_text.permute(1, 0, 2)
            encoded_text = self.transformer_encoder_layer(
                src=encoded_text, src_key_padding_mask=data['extra_mask'].bool())
            encoded_text = encoded_text.permute(1, 0, 2)
        infer_second_starts_logits = self.second_head_classifier(encoded_text)
        infer_second_starts = torch.sigmoid(infer_second_starts_logits)

        infer_second_ends_logits = self.second_tail_classifier(encoded_text)
        infer_second_ends = torch.sigmoid(infer_second_ends_logits)
        return infer_second_starts, infer_second_ends

    def get_text_embedding(self, token_ids, token_type_ids, input_mask, event_str_ids=None):
        encoded_text, globel_text_emb = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[:2]
        encoded_text = self.dropout(encoded_text)
        globel_text_emb = self.dropout(globel_text_emb)
        # if self.add_event_type:
        #     globel_text_emb = self.dropout(globel_text_emb).unsqueeze(
        #         1).repeat(1, encoded_text.shape[1], 1)
        #     concated = torch.cat((encoded_text, globel_text_emb), -1)
        #     f_gate = self.gate_cls(concated)
        #     encoded_text = torch.add(
        #         torch.mul(globel_text_emb, f_gate), encoded_text)
        # if self.add_event_type:
        #     if self.use_event_str:
        #         # encoded_text += globel_text_emb.unsqueeze(1)
        #         encoded_text = torch.cat((encoded_text, globel_text_emb.unsqueeze(1).repeat(
        #             1, encoded_text.shape[1], 1)), dim=-1)
        #     else:
        #         event_token_emb = self.event_type_embedding(event_str_ids)
        #         event_type_emb = self.event_trans(event_token_emb)

        #         # # (event_num, batch_size, 1, dim)
        #         # event_type_emb_ = event_type_emb.unsqueeze(
        #         #     -1).permute(1, 0, 3, 2)
        #         # globel_text_emb_ = globel_text_emb.unsqueeze(
        #         #     0).unsqueeze(-1)  # (1, batch_size, dim , 1)

        #         # scores = torch.matmul(
        #         #     event_type_emb_, globel_text_emb_).squeeze().permute(1, 0) / math.sqrt(self.encoder_config["hidden_size"])  # (batch_size, event_num)

        #         # probs = torch.softmax(scores, -1)

        #         # # (batch_size, event_num, dim)
        #         # event_emb = torch.mul(event_type_emb, probs.unsqueeze(-1))
        #         # event_emb = torch.sum(event_emb, dim=1)  # (batch_size, dim)

        #         event_emb = self.event_add_attention(
        #             globel_text_emb, event_type_emb)

        #         extended_event_emb = event_emb.unsqueeze(1).repeat(
        #             1, encoded_text.shape[1], 1)

        #         encoded_text = torch.cat(
        #             (encoded_text, extended_event_emb), dim=-1)
        # output = (encoded_text,)
        # if self.add_event_type:
        #     if self.use_event_str:
        #         output += (globel_text_emb,)
        #     else:
        #         output += (globel_text_emb, event_emb)
        output = (encoded_text, globel_text_emb)
        return output

    def event_emb_attention_encoder(self, data, cls_emb):
        event_token_emb = self.event_type_embedding(event_str_ids)
        event_type_emb = self.event_trans(event_token_emb)

    def first_label_extractor(self, encoded_text, globel_text_emb=None, data=None):
        if self.add_event_type:
            if self.use_event_str:
                event_emb = globel_text_emb
                encoded_text = torch.cat((encoded_text, event_emb.unsqueeze(1).repeat(
                    1, encoded_text.shape[1], 1)), dim=-1)
            else:
                # print(data['event_str_ids'].shape)
                # event_token_emb = self.event_type_embedding(
                #     data['event_str_ids'])
                # event_type_emb = self.event_trans(event_token_emb)

                # print(event_type_emb.shape)
                # event_emb = self.event_add_attention(
                #     globel_text_emb, event_type_emb)

                # extended_event_emb = event_emb.unsqueeze(1).repeat(
                #     1, encoded_text.shape[1], 1)

                # encoded_text = torch.cat(
                #     (encoded_text, extended_event_emb), dim=-1)

                # event_token_emb = self.event_type_embedding(
                #     data['event_str_ids'])
                # event_type_emb = self.event_trans(event_token_emb)

                # # (event_num, batch_size, 1, dim)
                # event_type_emb_ = event_type_emb.unsqueeze(
                #     -1).permute(1, 0, 3, 2)
                # globel_text_emb_ = globel_text_emb.unsqueeze(
                #     0).unsqueeze(-1)  # (1, batch_size, dim , 1)

                # scores = torch.matmul(
                #     event_type_emb_, globel_text_emb_).squeeze().permute(1, 0) / math.sqrt(self.encoder_config["hidden_size"])  # (batch_size, event_num)

                # probs = torch.softmax(scores, -1)

                # # (batch_size, event_num, dim)
                # event_emb = torch.mul(event_type_emb, probs.unsqueeze(-1))
                # event_emb = torch.sum(event_emb, dim=1)  # (batch_size, dim)

                # extended_event_emb = event_emb.unsqueeze(1).repeat(
                #     1, encoded_text.shape[1], 1)

                # encoded_text = torch.cat(
                #     (encoded_text, extended_event_emb), dim=-1)

                event_token_emb = self.event_type_embedding(
                    data['event_str_ids'])
                event_type_emb = self.event_trans(event_token_emb)

                extended_cls_emb = globel_text_emb.unsqueeze(
                    1).repeat(1, event_type_emb.shape[1], 1)

                event_type_emb_ = self.event_multi_head_attn_layer(
                    extended_cls_emb, event_type_emb, event_type_emb)[0]
                event_emb = torch.mean(event_type_emb_, dim=1)
                extended_event_emb = event_emb.unsqueeze(1).repeat(
                    1, encoded_text.shape[1], 1)

                encoded_text = torch.cat(
                    (encoded_text, extended_event_emb), dim=-1)

            infer_first_label_type, first_label_type_loss = self.first_label_type_classifier(
                data, event_emb)

        logits_first_starts = self.first_head_classifier(encoded_text)
        infer_first_starts = torch.sigmoid(logits_first_starts)

        logits_first_ends = self.first_tail_classifier(encoded_text)
        infer_first_ends = torch.sigmoid(logits_first_ends)

        output = (infer_first_starts, infer_first_ends)
        if self.add_event_type:
            output += (infer_first_label_type, first_label_type_loss)
        return output

    def first_label_type_classifier(self, data, cls_text_emb, is_event=False):
        first_label_type_logits = self.first_type_classifier(cls_text_emb)
        infer_label_type = torch.sigmoid(first_label_type_logits)
        output = (infer_label_type,)
        if data['first_types'] is not None:
            first_label_type_loss = nn.functional.binary_cross_entropy_with_logits(
                input=first_label_type_logits, target=data['first_types'])
            output = output + (first_label_type_loss,)
        return output

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def act_encoding(self, inputs, mask):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        x, (remainders, n_updates) = self.act_fc(x, inputs, mask.bool(), self.transformer_encoder_layer,
                                                 self.timing_signal, self.position_signal, self.encoder_config['num_hidden_layers'])
        # x = x.permute(1, 0, 2)
        return x, (remainders, n_updates)

    def _mal(self, a, b):
        if a.dim() == b.dim():
            return a * b
        elif a.dim() > b.dim():
            new_shape = b.size() + (1,) * (a.dim()-b.dim())
            return a * b.view(*new_shape)
        else:
            raise ValueError(
                "Wrong shape for a (shape {}) or b (shape {})".format(a.size, b.size))


class BertTriggerCrf(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertTriggerCrf, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.hidden_size = self.encoder_config['hidden_size']
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, args.first_label_num)
        self.crf = CRF(num_tags=args.first_label_num, batch_first=True)

    def forward(self, data,):
        encoded_text = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])[0]
        logits = self.classifier(encoded_text)
        outputs = (logits,)
        if data['first_starts'] is not None:
            loss = self.crf(emissions=logits,
                            tags=data['first_starts'], mask=data['input_mask'])
            outputs = (-1*loss,)+outputs
        return outputs

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        encoded_text = self.dropout(encoded_text)
        output = (encoded_text,)
        return output


class BertTriggerSoftmax(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertTriggerSoftmax, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.hidden_size = self.encoder_config['hidden_size']
        self.num_labels = args.first_label_num
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, args.first_label_num)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, data):
        encoded_text = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])[0]
        logits = self.classifier(encoded_text)

        active_loss = data['input_mask'].view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = data['first_starts'].view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)
        outputs = (loss, logits,)
        return outputs

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        encoded_text = self.dropout(encoded_text)
        output = (encoded_text,)
        return output


class BertSpan4ED(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertSpan4ED, self).__init__()

        self.pretrain_model_name = args.pretrain_model_name
        if "bert" == self.pretrain_model_name:
            self.bert = BertModel.from_pretrained(args.model_name_or_path)
            self.encoder_config = BertConfig.get_config_dict(args.model_name_or_path)[0]
        elif "roberta" == self.pretrain_model_name:
            self.bert = RobertaModel.from_pretrained(args.model_name_or_path)
            self.encoder_config = RobertaConfig.get_config_dict(args.model_name_or_path)[0]
        elif self.pretrain_model_name == "spanbert":
            self.bert = AutoModel.from_pretrained(args.model_name_or_path)
            self.encoder_config = BertConfig.get_config_dict(args.model_name_or_path)[0]

        self.hidden_size = self.encoder_config['hidden_size']
        self.dropout = nn.Dropout(args.dropout_rate)
        self.first_head_classifier = nn.Linear(
            (self.encoder_config["hidden_size"]), args.first_label_num)
        self.first_tail_classifier = nn.Linear(
            (self.encoder_config["hidden_size"]), args.first_label_num)
        # self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, data):
        first_output = self.get_text_embedding(data['token_ids'], data['token_type_ids']
                                               if self.pretrain_model_name == 'bert' else None, data['input_mask'])
        encoded_text = first_output[0]
        # encoded_text = self.fc_1(encoded_text)
        # encoded_text = torch.tanh(self.fc_2(encoded_text))
        infer_first_starts, infer_first_ends = self.first_label_extractor(
            encoded_text)

        output = (infer_first_starts, infer_first_ends)
        return output

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        if self.pretrain_model_name == 'bert':
            encoded_text = self.bert(input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        elif self.pretrain_model_name == 'roberta' or self.pretrain_model_name == 'spanbert':
            encoded_text = self.bert(input_ids=token_ids, attention_mask=input_mask)[0]
        encoded_text = self.dropout(encoded_text)
        output = (encoded_text,)
        return output

    def first_label_extractor(self, encoded_text):
        logits_first_starts = self.first_head_classifier(encoded_text)
        infer_first_starts = torch.sigmoid(logits_first_starts)

        logits_first_ends = self.first_tail_classifier(encoded_text)
        infer_first_ends = torch.sigmoid(logits_first_ends)
        return infer_first_starts, infer_first_ends


class BertTiggerExtractorWithLabel(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertTiggerExtractorWithLabel, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.hidden_size = self.encoder_config['hidden_size']
        self.use_auxiliary_task = args.use_auxiliary_task
        self.label_num = args.first_label_num
        self.dropout = nn.Dropout(args.dropout_rate)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        # self.first_head_classifier = nn.Linear(
        #     (self.encoder_config["hidden_size"]) * args.first_label_num, args.first_label_num)
        # self.first_tail_classifier = nn.Linear(
        #     (self.encoder_config["hidden_size"]) * args.first_label_num, args.first_label_num)
        self.first_head_classifier = Classifier_Layer(
            args.first_label_num, self.hidden_size)
        self.first_tail_classifier = Classifier_Layer(
            args.first_label_num, self.hidden_size)

        # self.first_head_classifier = nn.Linear(
        #     (self.encoder_config["hidden_size"]), 1)
        # self.first_tail_classifier = nn.Linear(
        #     (self.encoder_config["hidden_size"]), 1)

        if args.use_auxiliary_task:
            self.label_fusing_layer = Label_Attn_Fusion_Layer_for_token(
                self.hidden_size, args.first_label_num, label_emb_size=args.label_emb_size)
            self.first_type_classifier = nn.Linear(
                self.encoder_config["hidden_size"] * args.first_label_num, args.first_label_num)
        else:
            self.label_fusing_layer = Label_Fusion_Layer_for_Token(
                self.hidden_size, args.first_label_num, label_emb_size=args.label_emb_size)

        self.event_type_embedding = nn.Embedding(
            self.label_num, args.label_emb_size)
        self.event_type_embedding.weight.data.copy_(torch.from_numpy(
            np.load(args.glove_label_emb_file, allow_pickle=True)))
        # self.event_type_embedding.weight.requires_grad = False

    def forward(self, data, debug=False):
        first_output = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'], data['event_str_ids'], debug=debug)
        encoded_text = first_output[0]

        event_emb = self.event_type_embedding(data['event_str_ids'])

        # event_emb = self.get_text_embedding(
        #     data['label_token_ids'], data['label_token_type_ids'], data['label_input_mask'])[0]
        if self.use_auxiliary_task:
            second_output = self.label_fusing_layer(
                encoded_text, event_emb, return_label_embs=True)
        else:
            second_output = self.label_fusing_layer(
                encoded_text, event_emb, data['input_mask'])
        fused_feature = second_output[0]

        infer_first_starts, infer_first_ends = self.first_label_extractor(
            fused_feature, data['event_str_ids'])

        output = (infer_first_starts, infer_first_ends)

        if self.use_auxiliary_task:
            weighted_label_emb = second_output[1]
            if weighted_label_emb is not None:
                infer_label_type, first_label_type_loss = self.label_type_classifier(
                    data, weighted_label_emb, is_event=True)
                output += (infer_label_type, first_label_type_loss)
        return output

    def label_type_classifier(self, data, cls_text_emb, is_event=False):
        first_label_type_logits = self.first_type_classifier(cls_text_emb)
        infer_label_type = torch.sigmoid(first_label_type_logits)
        output = (infer_label_type,)
        if data['first_types'] is not None:
            first_label_type_loss = nn.functional.binary_cross_entropy_with_logits(
                input=first_label_type_logits, target=data['first_types'])
            output = output + (first_label_type_loss,)
        return output

    def get_text_embedding(self, token_ids, token_type_ids, input_mask, event_str_ids, debug=False):
        encoded_text, globel_text_emb = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[:2]
        encoded_text = self.dropout(encoded_text)

        globel_text_emb = self.dropout(globel_text_emb)
        output = (encoded_text, globel_text_emb)
        return output

    def first_label_extractor(self, encoded_text, event_str_ids):

        logits_first_starts = self.first_head_classifier(encoded_text)
        # logits_first_starts = self.first_head_classifier(
        #     encoded_text).squeeze()
        infer_first_starts = torch.sigmoid(logits_first_starts)

        logits_first_ends = self.first_tail_classifier(encoded_text)
        # logits_first_ends = self.first_tail_classifier(encoded_text).squeeze()
        infer_first_ends = torch.sigmoid(logits_first_ends)
        return infer_first_starts, infer_first_ends


class LEAR4ED(nn.Module):
    def __init__(self, args):
        super(LEAR4ED, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(args.model_name_or_path)[0]
        self.hidden_size = self.encoder_config['hidden_size']
        args.label_emb_size = self.encoder_config['hidden_size']
        self.use_attn = args.use_attn
        self.label_num = args.first_label_num
        self.dropout = nn.Dropout(args.dropout_rate)
        self.gradient_checkpointing = args.gradient_checkpointing

        # model layer
        self.bert = BertModel.from_pretrained(args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)
        self.first_head_classifier = Classifier_Layer(args.first_label_num, self.hidden_size)
        self.first_tail_classifier = Classifier_Layer(args.first_label_num, self.hidden_size)
        self.label_fusing_layer = Label_Fusion_Layer_for_Token(self.hidden_size, args.first_label_num, label_emb_size=args.label_emb_size)

    def forward(self, data, add_label_info=True, return_score=False, mode='train'):
        text_embs = self.get_text_embedding(data['token_ids'], data['token_type_ids'], data['input_mask'],
                                            return_token_level=True)['token_level_embs']
        if mode == 'train':
            label_embs = self.get_text_embedding(data['label_token_ids'], data['label_token_type_ids'], data['label_input_mask'], return_token_level=True)['token_level_embs'] if self.use_attn else self.get_text_embedding(
                data['label_token_ids'], data['label_token_type_ids'], data['label_input_mask'], return_sentence_level=True)['sentence_level_embs']  # [class_num, hidden_dim]
        else:
            label_embs = data['label_embs']
        if not add_label_info:
            label_embs = label_embs.detach()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, use_attn=self.use_attn, return_scores=return_score)
            return custom_forward

        if self.gradient_checkpointing:
            fused_results = torch.utils.checkpoint.checkpoint(create_custom_forward(
                self.label_fusing_layer), text_embs, label_embs, data['input_mask'], data['label_input_mask'])
        else:
            fused_results = self.label_fusing_layer(
                text_embs, label_embs, data['input_mask'], data['label_input_mask'], use_attn=self.use_attn, return_scores=return_score)

        # [bs, seq_len, class_num, hidden_size]
        fused_feature = fused_results[0]
        infer_first_starts, infer_first_ends = self.first_label_extractor(fused_feature)
        output = (infer_first_starts, infer_first_ends)
        if return_score:
            output += (fused_results[-1],)
        return output

    def get_text_embedding(self, token_ids, token_type_ids, input_mask, return_token_level=False, return_sentence_level=False):
        token_level_embs, sentence_level_embs = self.bert(input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[:2]
        output = {}
        if return_token_level:
            token_level_embs = self.dropout(token_level_embs)
            output['token_level_embs'] = token_level_embs
        if return_sentence_level:
            sentence_level_embs = self.dropout(sentence_level_embs)
            output['sentence_level_embs'] = sentence_level_embs
        return output

    def first_label_extractor(self, encoded_text):

        if self.gradient_checkpointing:
            logits_first_starts = torch.utils.checkpoint.checkpoint(
                self.first_head_classifier, encoded_text)
        else:
            logits_first_starts = self.first_head_classifier(encoded_text)
        infer_first_starts = torch.sigmoid(logits_first_starts)

        if self.gradient_checkpointing:
            logits_first_ends = torch.utils.checkpoint.checkpoint(
                self.first_tail_classifier, encoded_text)
        else:
            logits_first_ends = self.first_tail_classifier(encoded_text)
        infer_first_ends = torch.sigmoid(logits_first_ends)
        return infer_first_starts, infer_first_ends
