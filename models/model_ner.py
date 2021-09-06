
import torch
from torch import nn
from transformers import BertConfig, BertModel
from torch.autograd import Variable
from utils.crf import CRF
import numpy as np
import utils.model_utils as model_utils


class BertNerCrf(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertNerCrf, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.hidden_size = self.encoder_config['hidden_size']
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, args.first_label_num)
        self.crf = CRF(num_tags=args.first_label_num, batch_first=True)

    def forward(self, data):
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


class BertNerSoftmax(nn.Module):
    def __init__(self, args, input_dropout=0.0):
        super(BertNerSoftmax, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.hidden_size = self.encoder_config['hidden_size']
        self.num_labels = args.first_label_num
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, args.first_label_num)
        # self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fct = nn.CrossEntropyLoss()

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


class BertNerSpan(nn.Module):
    def __init__(self, args):
        super(BertNerSpan, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.hidden_size = self.encoder_config["hidden_size"]
        self.dropout = nn.Dropout(0.1)
        self.label_num = args.first_label_num
        self.bert = BertModel.from_pretrained(
            args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)

        # self.entity_start_classifier = nn.Linear(
        #     self.encoder_config["hidden_size"], self.label_num)
        # self.entity_end_classifier = nn.Linear(
        #     self.encoder_config["hidden_size"], self.label_num)
        self.entity_start_classifier = model_utils.Classifier_Layer(
            self.label_num, self.hidden_size)
        self.entity_end_classifier = model_utils.Classifier_Layer(
            self.label_num, self.hidden_size)

    def forward(self, data):
        encoded_text = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])  # [bs,seqlen,hidden_dim]

        encoded_text = encoded_text.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)  # [bs,seqlen,class_num, hidden_dim]

        logits_start = self.entity_start_classifier(
            encoded_text)  # [bs,seqlen,class_num]
        # infer_start = torch.sigmoid(logits_start)

        logits_end = self.entity_end_classifier(encoded_text)
        # infer_end = torch.sigmoid(logits_end)
        # return infer_start, infer_end
        return logits_start, logits_end

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        encoded_text = self.dropout(encoded_text)
        return encoded_text


class BertNerSpanMatrix(nn.Module):
    def __init__(self, args):
        super(BertNerSpanMatrix, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.dropout = nn.Dropout(0.1)
        self.label_num = args.first_label_num
        self.hidden_size = self.encoder_config["hidden_size"]
        self.gradient_checkpointing = args.gradient_checkpointing
        self.bert = BertModel.from_pretrained(
            args.model_name_or_path, gradient_checkpointing=self.gradient_checkpointing)
        self.entity_start_classifier = nn.Linear(
            self.encoder_config["hidden_size"], self.label_num)
        self.entity_end_classifier = nn.Linear(
            self.encoder_config["hidden_size"], self.label_num)
        self.span_embedding = model_utils.MultiNonLinearClassifier(
            self.hidden_size * 2, self.label_num, args.classifier_dropout_rate)

    def forward(self, data):
        encoded_text = self.get_text_embedding(
            data['token_ids'], data['token_type_ids'], data['input_mask'])  # [batch_size, seq_len, hidden_dim]
        seq_len = encoded_text.shape[1]

        if self.gradient_checkpointing:
            logits_start = torch.utils.checkpoint.checkpoint(
                self.entity_start_classifier, encoded_text)
            logits_end = torch.utils.checkpoint.checkpoint(
                self.entity_end_classifier, encoded_text)
        else:
            logits_start = self.entity_start_classifier(encoded_text)
            logits_end = self.entity_end_classifier(encoded_text)

        start_extend = encoded_text.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = encoded_text.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat((start_extend, end_extend), dim=-1)

        # [batch_size, seq_len, seq_len, class_num]
        if self.gradient_checkpointing:
            span_logits = torch.utils.checkpoint.checkpoint(
                self.span_embedding, span_matrix)
        else:
            span_logits = self.span_embedding(span_matrix)

        # return infer_start, infer_end, span_logits
        return logits_start, logits_end, span_logits

    def get_text_embedding(self, token_ids, token_type_ids, input_mask):
        encoded_text = self.bert(
            input_ids=token_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        encoded_text = self.dropout(encoded_text)
        return encoded_text


class LEARNer4Flat(nn.Module):
    def __init__(self, args):
        super(LEARNer4Flat, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.dropout = nn.Dropout(args.dropout_rate)
        self.label_num = args.first_label_num
        self.use_attn = args.use_attn
        self.use_random_label_emb = args.use_random_label_emb
        self.average_pooling = args.average_pooling
        self.do_add = args.do_add
        self.use_label_embedding = args.use_label_embedding
        self.hidden_size = self.encoder_config['hidden_size']
        self.gradient_checkpointing = args.gradient_checkpointing
        self.bert = BertModel.from_pretrained(args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)
        self.entity_start_classifier = model_utils.Classifier_Layer(self.label_num, self.hidden_size)
        self.entity_end_classifier = model_utils.Classifier_Layer(self.label_num, self.hidden_size)
        self.label_fusing_layer = model_utils.Label_Fusion_Layer_for_Token(
            self.encoder_config["hidden_size"], self.label_num, 200 if self.use_label_embedding else self.hidden_size)
        if self.use_label_embedding:
            self.label_ann_vocab_size = args.label_ann_vocab_size
            self.label_embedding_layer = nn.Embedding(args.label_ann_vocab_size, 200)
            glove_embs = torch.from_numpy(np.load(args.glove_label_emb_file, allow_pickle=True)).to(args.device)
            self.label_embedding_layer.weight.data.copy_(glove_embs)

    def forward(self, data, label_token_ids, label_token_type_ids, label_input_mask, add_label_info=True, return_score=False, mode='train', return_bert_attention=False):
        results = self.bert(input_ids=data['token_ids'], token_type_ids=data['token_type_ids'],
                            attention_mask=data['input_mask'], output_attentions=return_bert_attention)
        encoded_text = results[0]
        encoded_text = self.dropout(encoded_text)

        # batch_size, seq_len = encoded_text.shape[:2]
        if self.use_label_embedding:
            label_embs = self.label_embedding_layer(label_token_ids)
        elif self.use_random_label_emb:
            label_embs = data['random_label_emb']
        else:
            label_embs = self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids, attention_mask=label_input_mask)[
                0] if self.use_attn else self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids, attention_mask=label_input_mask)[1]
        if not add_label_info:
            label_embs = label_embs.detach()  # only stop gradient of current step. It will update according to history if adam used.

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, return_scores=return_score, use_attn=self.use_attn, do_add=self.do_add, average_pooling=self.average_pooling)
            return custom_forward

        if mode == 'train' and self.gradient_checkpointing:
            fused_results = torch.utils.checkpoint.checkpoint(create_custom_forward(
                self.label_fusing_layer), encoded_text, label_embs, data['input_mask'], label_input_mask)
        else:
            fused_results = self.label_fusing_layer(
                encoded_text, label_embs, data['input_mask'], label_input_mask=label_input_mask, return_scores=return_score, use_attn=self.use_attn, do_add=self.do_add, average_pooling=self.average_pooling)

        fused_feature = fused_results[0]

        if mode == 'train' and self.gradient_checkpointing:
            # logits_start = torch.utils.checkpoint.checkpoint(
            #     self.entity_start_classifier, fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            # logits_end = torch.utils.checkpoint.checkpoint(
            #     self.entity_end_classifier, fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            logits_start = torch.utils.checkpoint.checkpoint(self.entity_start_classifier, fused_feature)
            logits_end = torch.utils.checkpoint.checkpoint(self.entity_end_classifier, fused_feature)
        else:
            # logits_start = self.entity_start_classifier(
            #     fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            # logits_end = self.entity_end_classifier(
            #     fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            logits_start = self.entity_start_classifier(fused_feature)
            logits_end = self.entity_end_classifier(fused_feature)
        # infer_start = torch.sigmoid(logits_start)
        # infer_end = torch.sigmoid(logits_end)
        output = (logits_start, logits_end)
        if return_score:
            output += (fused_results[-1],)
        if return_bert_attention:
            output += (results[-1],)
        return output


class LEARNer4Nested(nn.Module):
    def __init__(self, args):
        super(LEARNer4Nested, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        self.dropout = nn.Dropout(args.dropout_rate)
        self.label_num = args.first_label_num
        self.hidden_size = self.encoder_config["hidden_size"]
        self.use_attn = args.use_attn
        self.average_pooling = args.average_pooling
        self.use_random_label_emb = args.use_random_label_emb
        self.do_add = args.do_add
        self.use_label_embedding = args.use_label_embedding
        self.gradient_checkpointing = args.gradient_checkpointing
        self.bert = BertModel.from_pretrained(args.model_name_or_path, gradient_checkpointing=self.gradient_checkpointing)
        self.span_embedding = model_utils.MultiNonLinearClassifierForMultiLabel(self.hidden_size * 2, self.label_num, args.classifier_dropout_rate)
        self.label_fusing_layer = model_utils.Label_Fusion_Layer_for_Token(
            self.hidden_size, self.label_num, 300 if self.use_label_embedding else self.hidden_size)
        self.entity_start_classifier = model_utils.Classifier_Layer(self.label_num, self.hidden_size)
        self.entity_end_classifier = model_utils.Classifier_Layer(self.label_num, self.hidden_size)
        if self.use_label_embedding:
            self.label_ann_vocab_size = args.label_ann_vocab_size
            self.label_embedding_layer = nn.Embedding(args.label_ann_vocab_size, 300)
            glove_embs = torch.from_numpy(np.load(args.glove_label_emb_file, allow_pickle=True)).to(args.device)
            self.label_embedding_layer.weight.data.copy_(glove_embs)

    def forward(self, data, label_token_ids, label_token_type_ids, label_input_mask, add_label_info=True, return_score=False, mode='train', return_bert_attention=False):

        # text_emb = self.bert(input_ids=data['token_ids'], token_type_ids=data['token_type_ids'],
        #                      attention_mask=data['input_mask'])[0]  # [batch_size, seq_len, hidden_dim]
        text_emb = self.get_text_embedding(data['token_ids'], data['token_type_ids'], data['input_mask'], return_token_level=True)['token_level_embs']
        seq_len = text_emb.shape[1]

        if self.use_label_embedding:
            label_embs = self.label_embedding_layer(label_token_ids)
        elif self.use_random_label_emb:
            label_embs = data['random_label_emb']
        else:
            # label_embs = self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids,
            #                        attention_mask=label_input_mask)[0]
            label_embs = self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids, attention_mask=label_input_mask)[
                0] if self.use_attn else self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids, attention_mask=label_input_mask)[1]

        # if not add_label_info:
        #     label_embs = label_embs.detach()
        # if add_label_info:
        #     label_embs = label_embs.detach()
        # [batch_size, seq_len, class_num, hidden_dim]

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, use_attn=self.use_attn, do_add=self.do_add, average_pooling=self.average_pooling)
            return custom_forward

        if mode == 'train' and self.gradient_checkpointing:
            fused_results = torch.utils.checkpoint.checkpoint(create_custom_forward(
                self.label_fusing_layer), text_emb, label_embs, data['input_mask'], label_input_mask)
        else:
            fused_results = self.label_fusing_layer(
                text_emb, label_embs, data['input_mask'], label_input_mask, use_attn=self.use_attn, do_add=self.do_add, average_pooling=self.average_pooling)
        fused_emb = fused_results[0]
        if mode == 'train' and self.gradient_checkpointing:
            logits_start = torch.utils.checkpoint.checkpoint(
                self.entity_start_classifier, fused_emb)
            logits_end = torch.utils.checkpoint.checkpoint(
                self.entity_end_classifier, fused_emb)
        else:
            logits_start = self.entity_start_classifier(fused_emb)
            logits_end = self.entity_end_classifier(fused_emb)

        # [batch_size, seq_len, seq_len, class_num, hidden_dim]
        start_extend = fused_emb.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        end_extend = fused_emb.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        span_matrix = torch.cat((start_extend, end_extend), dim=-1)

        # [batch_size, seq_len, seq_len, class_num]
        if mode == 'train' and self.gradient_checkpointing:
            span_logits = torch.utils.checkpoint.checkpoint(
                self.span_embedding, span_matrix)
        else:
            span_logits = self.span_embedding(span_matrix)

        return (logits_start, logits_end, span_logits)

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
