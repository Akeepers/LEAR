from os import EX_NOHOST
import torch
import math
from torch import nn
from torch.nn.parameter import Parameter
# from transformers impo
import numpy as np
import torch.nn.functional as F


class Classifier_Layer(nn.Module):
    def __init__(self, class_num, out_features, bias=True):
        super(Classifier_Layer, self).__init__()
        self.class_num = class_num
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(class_num, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(class_num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     for i in range(self.class_num):
    #         self.weight[i].data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         for i in range(self.class_num):
    #             self.bias[i].data.uniform_(-stdv, stdv)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # batch_size, seq_len = input.shape[:2]
        # input = input.contiguous().view(batch_size * seq_len,
        #                                 self.class_num, self.out_features)
        # x = input * self.weight  # [-1, class_num, dim]
        # print(input.shape)
        # print(self.weight.shape)
        x = torch.mul(input, self.weight)
        # (class_num, 1)
        x = torch.sum(x, -1)  # [-1, class_num]
        if self.bias is not None:
            x = x + self.bias
        # x = x.contiguous().view(batch_size, seq_len,
        #                         self.class_num)
        return x

    def extra_repr(self):
        return 'class_num={}, out_features={}, bias={}'.format(
            self.class_num, self.out_features, self.bias is not None)


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate=0.3):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)

        # features_output2 = self.classifier2(input_features)
        return features_output2


class MultiNonLinearClassifierForMultiLabel(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifierForMultiLabel, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = Classifier_Layer(num_label, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)

        # features_output1 = F.gelu(input_features)
        # features_output1 = self.dropout(features_output1)

        features_output2 = self.classifier2(features_output1)

        # features_output2 = self.classifier2(input_features)
        return features_output2


class GGNN_1(nn.Module):
    def __init__(self, input_dim, time_step, in_matrix=None, out_matrix=None):
        super(GGNN_1, self).__init__()
        self.input_dim = input_dim
        self.time_step = time_step
        self._in_matrix = in_matrix
        self._out_matrix = out_matrix

        self.fc_eq3_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq3_u = nn.Linear(input_dim, input_dim)
        self.fc_eq4_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq4_u = nn.Linear(input_dim, input_dim)
        self.fc_eq5_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq5_u = nn.Linear(input_dim, input_dim)

    def forward(self, input, batch_in_matrix=None, batch_out_matrix=None):
        batch_size = input.size()[0]
        # [bs, num_classes, image_feature_dim] -> [bs * num_classes, image_feature_dim]
        input = input.view(-1, self.input_dim)
        node_num = self._in_matrix.size()[0]

        # node_num = batch_in_matrix.shape[1]
        batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)
        batch_in_matrix = self._in_matrix.repeat(
            batch_size, 1).view(batch_size, node_num, -1)
        batch_out_matrix = self._out_matrix.repeat(
            batch_size, 1).view(batch_size, node_num, -1)
        for t in range(self.time_step):
            # eq(2)
            av = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(
                batch_out_matrix, batch_aog_nodes)), 2)
            av = av.view(batch_size * node_num, -1)

            flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1)

            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(
                av) + self.fc_eq3_u(flatten_aog_nodes))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(
                av) + self.fc_eq3_u(flatten_aog_nodes))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(
                av) + self.fc_eq5_u(rv * flatten_aog_nodes))

            flatten_aog_nodes = (1 - zv) * flatten_aog_nodes + zv * hv
            batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)
        return batch_aog_nodes


class GGNN_2(nn.Module):
    def __init__(self, input_dim, time_step,  in_matrix=None, out_matrix=None):
        super(GGNN_2, self).__init__()
        self.input_dim = input_dim
        self.time_step = time_step
        # self._in_matrix = in_matrix
        # self._out_matrix = out_matrix

        self.fc_eq3_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq3_u = nn.Linear(input_dim, input_dim)
        self.fc_eq4_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq4_u = nn.Linear(input_dim, input_dim)
        self.fc_eq5_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq5_u = nn.Linear(input_dim, input_dim)

    def forward(self, input, batch_in_matrix=None, batch_out_matrix=None):
        batch_size = input.size()[0]
        # [bs, num_classes, image_feature_dim] -> [bs * num_classes, image_feature_dim]
        input = input.view(-1, self.input_dim)
        # node_num = self._in_matrix.size()[0]

        node_num = batch_in_matrix.shape[1]
        batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)
        # batch_in_matrix = self._in_matrix.repeat(
        #     batch_size, 1).view(batch_size, node_num, -1)
        # batch_out_matrix = self._out_matrix.repeat(
        #     batch_size, 1).view(batch_size, node_num, -1)
        for t in range(self.time_step):
            # eq(2)
            av = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(
                batch_out_matrix, batch_aog_nodes)), 2)
            av = av.view(batch_size * node_num, -1)

            flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1)

            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(
                av) + self.fc_eq3_u(flatten_aog_nodes))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(
                av) + self.fc_eq3_u(flatten_aog_nodes))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(
                av) + self.fc_eq5_u(rv * flatten_aog_nodes))

            flatten_aog_nodes = (1 - zv) * flatten_aog_nodes + zv * hv
            batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)
        return batch_aog_nodes


class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, mask, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        # [B, S]
        halting_probability = torch.zeros(
            inputs.shape[0], inputs.shape[1]).cuda()
        # [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        # [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        # [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while(((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:,
                                     :inputs.shape[1], :].type_as(inputs.data)
            state = state + \
                pos_enc[:, step, :].unsqueeze(1).repeat(
                    1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running >
                          self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <=
                             self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(encoder_output):
                state, _ = fn((state, encoder_output))
            else:
                # apply transformation on the state
                state = state.permute(1, 0, 2)
                state = fn(state, src_key_padding_mask=mask)
                state = state.permute(1, 0, 2)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) +
                              (previous_state * (1 - update_weights.unsqueeze(-1))))
            # previous_state is actually the new_state at end of hte loop
            # to save a line I assigned to previous_state so in the next
            # iteration is correct. Notice that indeed we return previous_state
            step += 1
        return previous_state, (remainders, n_updates)


class Label_Fusion_Layer_for_Token(nn.Module):
    def __init__(self, hidden_size, label_num, label_emb_size=300):
        super(Label_Fusion_Layer_for_Token, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_2 = nn.Linear(label_emb_size, self.hidden_size, bias=False)
        # self.fc_3 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc_3 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        # self.fc_4 = nn.Linear(self.hidden_size, 1)
        self.fc_5 = nn.Linear(self.hidden_size, self.hidden_size)

        # self.fc_6 = nn.Linear(self.label_num, label_num, bias=False)

    def forward(self, token_embs, label_embs, input_mask, label_input_mask=None, return_scores=False, use_attn=False, do_add=False, average_pooling=False):
        # [bs, seq_len, hidden_size]; [bs, label_num, 300]

        # batch_size, seq_len = token_embs.shape[:2]
        # token_features = token_embs.contiguous().view(
        #     batch_size*seq_len, -1)  # [bs * seq_len, hidden_dim]
        # token_features = self.fc_1(token_features).unsqueeze(1).repeat(
        #     1, self.label_num, 1)  # [bs * seq_len, label_num, hidden_dim]

        # label_features = self.fc_2(label_embs).unsqueeze(
        #     1).repeat(1, seq_len, 1, 1).view(batch_size*seq_len, self.label_num, self.hidden_size)  # [bs * seq_len, label_num, hidden_dim]

        # temp_features = self.fc_3(torch.tanh(
        #     token_features*label_features)).contiguous().view(batch_size, seq_len, -1)  # [bs * seq_len * label_num, hidden_dim]
        # return (temp_features,)
        # return self.get_fused_feature_with_cat(token_embs, label_embs)
        # return self.get_fused_feature_with_cos_weight(token_embs, label_embs, input_mask, return_scores=return_scores)

        # print(use_attn)
        if use_attn:
            if average_pooling:
                return self.get_fused_feature_with_average_pooling(token_embs, label_embs, input_mask, label_input_mask)
            else:
                return self.get_fused_feature_with_attn(token_embs, label_embs, input_mask, label_input_mask, return_scores=return_scores)
        elif do_add:
            return self.get_fused_feature_with_add(token_embs, label_embs, input_mask)
        else:
            return self.get_fused_feature_with_cos_weight(token_embs, label_embs, input_mask, return_scores=return_scores)
        # return self.get_fused_feature_with_biliear(token_embs, label_embs)

    def get_fused_feature_1(self, token_feature, label_feature):
        batch_size, seq_len = token_feature.shape[:2]
        token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
            1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        label_features = self.fc_2(label_feature).unsqueeze(
            1).repeat(1, seq_len, 1, 1)

        fused_features = token_features + label_features
        fused_features = torch.tanh(self.fc_3(fused_features))
        # print(fused_features.shape)
        return (fused_features.contiguous().view(batch_size, seq_len, -1),)

    def get_fused_feature_with_add(self, token_feature, label_feature, input_mask, return_scores=False):
        batch_size, seq_len = token_feature.shape[:2]
        if len(token_feature.shape) != len(input_mask.shape):
            input_mask = input_mask.unsqueeze(-1)

        token_feature = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        label_feature = label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        token_feature = token_feature.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)

        fused_feature = token_feature + label_feature

        output = (fused_feature,)

        return output

    def get_fused_feature_with_cat(self, token_feature, label_feature):
        batch_size, seq_len = token_feature.shape[:2]
        token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
            1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        label_features = self.fc_2(label_feature).unsqueeze(
            1).repeat(1, seq_len, 1, 1)

        fused_features = torch.cat((token_features, label_features), dim=-1)
        # print(fused_features.shape)
        fused_features = torch.tanh(self.fc_3(fused_features))
        # print(fused_features.shape)
        return (fused_features.contiguous().view(batch_size, seq_len, -1),)

    def get_fused_feature_with_weight(self, token_feature, label_feature):
        batch_size, seq_len = token_feature.shape[:2]
        # token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
        #     1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        # label_features = self.fc_2(label_feature).unsqueeze(
        #     1).repeat(1, seq_len, 1, 1)
        token_features = token_feature.unsqueeze(2).repeat(
            1, 1, self.label_num, 1)
        label_features = label_feature.unsqueeze(
            1).repeat(1, seq_len, 1, 1)

        temp_features = self.fc_3(torch.tanh(token_features * label_features))
        temp_features = self.fc_4(temp_features)
        scores = torch.sigmoid(temp_features.permute(
            0, 1, 3, 2)).permute(0, 1, 3, 2)
        weighted_label_features = scores * label_features
        fused_features = token_features + weighted_label_features
        # fused_features = token_features *scores
        # fused_features = token_features
        fused_features = torch.tanh(self.fc_5(fused_features))
        # return (fused_features.contiguous().view(batch_size, seq_len, -1),)
        return (fused_features, scores.squeeze(-1))

    def get_fused_feature_with_cos_weight_for_batch(self, token_feature, label_feature, input_mask):
        batch_size, seq_len = token_feature.shape[:2]
        if len(token_feature.shape) != len(input_mask.shape):
            input_mask = input_mask.unsqueeze(-1)

        token_feature = self.fc_1(token_feature).unsqueeze(
            2).repeat(1, 1, self.label_num, 1)
        label_feature = self.fc_2(label_feature).unsqueeze(
            1).repeat(1, seq_len, 1, 1)
        # token_feature_masked = token_feature * input_mask
        # token_feature_masked = token_feature_masked.unsqueeze(
        #     2).repeat(1, 1, self.label_num, 1)
        token_feature_norm = nn.functional.normalize(
            token_feature, p=2, dim=-1)

        label_feature_norm = nn.functional.normalize(
            label_feature, p=2, dim=-1)
        # [bs, seq_len, clas_num]
        scores = token_feature_norm * label_feature_norm  # cosine-sim
        scores = torch.sum(scores, dim=-1)
        scores = torch.relu(scores).unsqueeze(-1)
        # label_feature = label_feature.unsqueeze(
        #     0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        # token_feature = token_feature.unsqueeze(
        #     2).repeat(1, 1, self.label_num, 1)
        weighted_label_feature = scores * label_feature
        # print(token_feature.shape)
        # print(weighted_label_feature.shape)
        fused_feature = token_feature + weighted_label_feature
        # fused_feature = torch.cat(
        #     (token_feature, weighted_label_feature), dim=-1)
        fused_feature = torch.tanh(self.fc_5(fused_feature))
        return (fused_feature, scores.squeeze(-1))

    def get_fused_feature_with_cos_weight(self, token_feature, label_feature, input_mask, return_scores=False):
        batch_size, seq_len = token_feature.shape[:2]
        if len(token_feature.shape) != len(input_mask.shape):
            input_mask = input_mask.unsqueeze(-1)

        token_feature = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        # token_feature = self.dropout(token_feature)
        # label_feature = self.dropout(label_feature)

        token_feature_masked = token_feature * input_mask
        token_feature_norm = nn.functional.normalize(
            token_feature_masked, p=2, dim=-1)

        label_feature_t = label_feature.permute(
            1, 0)  # [hidden_dim, class_num]
        # label_feature_t = label_feature.transpose(0, 1)
        label_feature_t_norm = nn.functional.normalize(
            label_feature_t, p=2, dim=0)
        # [bs, seq_len, clas_num]
        scores = torch.matmul(token_feature_norm,
                              label_feature_t_norm).unsqueeze(-1)  # cosine-sim

        # scores = torch.relu(scores)

        # scores = scores.unsqueeze(-1)

        # label_feature = label_feature.unsqueeze(
        #     0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        # token_feature = token_feature.unsqueeze(
        #     2).expand(-1, -1, self.label_num, -1)
        label_feature = label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        token_feature = token_feature.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)
        weighted_label_feature = scores * label_feature
        # print(token_feature.shape)
        # print(weighted_label_feature.shape)
        fused_feature = token_feature + weighted_label_feature
        # fused_feature = torch.cat(
        #     (token_feature, weighted_label_feature), dim=-1)

        # fused_feature = torch.tanh(self.fc_5(fused_feature))
        # fused_feature = self.fc_6(
        #     fused_feature.transpose(2, 3)).transpose(2, 3)

        output = (fused_feature,)

        if return_scores:
            output += (scores.squeeze(-1),)
        return output

    def get_fused_feature_with_attn(self, token_feature, label_feature, input_mask, label_input_mask, return_scores=False):
        """
            token_feature: [batch_size, context_seq_len, hidden_dim]
            label_feature: [class_num, label_seq_len, hidden_dim]
        """
        # print('attn')
        batch_size, context_seq_len = token_feature.shape[:2]

        token_feature_fc = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        # [hidden_dim, class_num, label_seq_len]
        label_feature_t = label_feature.permute(
            2, 0, 1).view(self.hidden_size, -1)
        # label_feature_t_norm = nn.functional.normalize(
        #     label_feature_t, p=2, dim=0)

        # [bs, context_seq_len, class_num, label_seq_len]
        # print(token_feature.shape)
        # print(label_feature_t.shape)
        scores = torch.matmul(token_feature_fc, label_feature_t).view(
            batch_size, context_seq_len, self.label_num, -1)  # dot

        # scores = scores / math.sqrt(self.hidden_size)

        extended_mask = label_input_mask[None, None, :, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        scores = scores + extended_mask

        # [bs, context_seq_len, class_num, label_seq_len]
        scores = torch.softmax(scores, dim=-1)

        # torch.save(scores.to(torch.device('cpu')),'/data/yangpan/workspace/research/SERS/visualization/attn.pt')

        # [bs, context_seq_len, class_num, label_seq_len, hidden_dim]
        weighted_label_feature = label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, context_seq_len, 1, 1, 1) * scores.unsqueeze(-1)

        token_feature_fc = token_feature_fc.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)
        # token_feature = token_feature.unsqueeze(2).repeat(1, 1, self.label_num, 1)
        # [bs, context_seq_len, class_num, hidden_dim]
        weighted_label_feature_sum = torch.sum(weighted_label_feature, dim=-2)

        # [bs, context_seq_len, class_num, hidden_dim]
        fused_feature = token_feature_fc + weighted_label_feature_sum
        # fused_feature = token_feature + weighted_label_feature_sum
        # fused_feature = torch.cat(
        #     (token_feature, weighted_label_feature), dim=-1)

        fused_feature = torch.tanh(self.fc_5(fused_feature))
        # fused_feature = self.fc_6(
        #     fused_feature.transpose(2, 3)).transpose(2, 3)
        output = (fused_feature,)

        if return_scores:
            # print(scores.shape)
            output += (scores.squeeze(-1),)
        return output

    def get_fused_feature_with_average_pooling(self, token_feature, label_feature, input_mask, label_input_mask, return_scores=False):
        """
            token_feature: [batch_size, context_seq_len, hidden_dim]
            label_feature: [class_num, label_seq_len, hidden_dim]
        """
        batch_size, context_seq_len = token_feature.shape[:2]

        token_feature_fc = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        # [class_num, hidden_dim]
        averaged_label_feature = label_feature.mean(dim=1)

        # [bs, context_seq_len, class_num, hidden_dim]
        weighted_label_feature = averaged_label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, context_seq_len, 1, 1)

        token_feature_fc = token_feature_fc.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)

        # [bs, context_seq_len, class_num, hidden_dim]
        fused_feature = token_feature_fc + weighted_label_feature

        fused_feature = torch.tanh(self.fc_5(fused_feature))
        output = (fused_feature,)

        return output

    def get_fused_feature_with_biliear(self, token_feature, label_feature):
        batch_size, seq_len = token_feature.shape[:2]
        token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
            1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        label_features = self.fc_2(label_feature).unsqueeze(
            1).repeat(1, seq_len, 1, 1)

        fused_features = self.fc_3(torch.tanh(token_features * label_features))
        # fused_features = torch.tanh(self.fc_5(fused_features))
        # return (fused_features.contiguous().view(batch_size, seq_len, -1),)
        return (fused_features,)

    def get_fused_feature(self, token_feature, label_feature):
        batch_size, seq_len = token_feature.shape[:2]
        token_features = token_embs.contiguous().view(
            batch_size*seq_len, -1)  # [bs * seq_len, hidden_dim]
        token_features = self.fc_1(token_feature).unsqueeze(1).repeat(
            1, self.label_num, 1)  # [bs * seq_len, label_num, hidden_dim]

        label_features = self.fc_2(label_embs).unsqueeze(
            1).repeat(1, seq_len, 1, 1).view(batch_size*seq_len, self.label_num, self.hidden_size)  # [bs * seq_len, label_num, hidden_dim]

        temp_features = self.fc_3(torch.tanh(
            token_features*label_features)).contiguous().view(batch_size, seq_len, -1)  # [bs * seq_len * label_num, hidden_dim]
        return (temp_features,)


class Label_Fusion_Layer_for_Classification(nn.Module):
    def __init__(self, hidden_size, label_num, label_emb_size=300):
        super(Label_Fusion_Layer_for_Classification, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_2 = nn.Linear(label_emb_size, self.hidden_size, bias=False)
        self.fc_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_a = nn.Linear(self.hidden_size, 1)
        self.fc_5 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token_embs, label_embs, return_label_embs=False, input_mask=None):
        # [bs, seq_len, hidden_size]; [bs, label_num, 300]

        return self.get_fused_feature_with_cos(token_embs, label_embs, input_mask)

    def get_fused_feature_with_cos(self, token_feature, label_feature, input_mask=None):

        if len(token_feature.shape) != len(input_mask.shape):
            input_mask = input_mask.unsqueeze(-1)

        token_feature_masked = token_feature * input_mask
        token_feature_norm = nn.functional.normalize(
            token_feature_masked, p=2, dim=-1)

        label_feature_t = label_feature.permute(
            1, 0)  # [hidden_dim, class_num]
        label_feature_t_norm = nn.functional.normalize(label_feature_t, dim=0)

        # [bs, seq_len, clas_num]
        scores = torch.matmul(token_feature_norm,
                              label_feature_t_norm)  # cosine-sim

        scores = torch.relu(scores)
        scores = torch.max(scores, dim=-1, keepdim=True)[0]

        scores = scores * input_mask
        scores = torch.softmax(scores, dim=1)
        weighted_token_feature = token_feature * scores
        seq_feature = torch.sum(
            weighted_token_feature, dim=1) / torch.sum(input_mask, dim=1)
        return (seq_feature,)

    def get_fused_feature_with_weight_softmax(self, token_feature, label_feature, input_mask=None):
        batch_size, seq_len = token_feature.shape[:2]
        token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
            1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        label_features = self.fc_2(label_feature).unsqueeze(
            1).repeat(1, seq_len, 1, 1)

        # [bs , seq_len, label_num, hidden_dim]
        temp_features = self.fc_3(torch.tanh(token_features * label_features))
        scores = self.fc_a(temp_features)
        # scores = torch.softmax(scores, dim=2)
        scores = torch.softmax(scores, dim=1)
        weighted_token_features = token_features * scores

        extended_mask = input_mask.unsqueeze(-1).unsqueeze(-1)
        weighted_token_features = torch.mul(
            weighted_token_features, extended_mask)

        # [bs, label_num, hidden_dim]
        seq_features = torch.sum(
            weighted_token_features, dim=1) / torch.sum(extended_mask, dim=1)
        return (seq_features,)

    def get_fused_feature_with_weight_sigmoid(self, token_feature, label_feature):
        batch_size, seq_len = token_feature.shape[:2]
        token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
            1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        label_features = self.fc_2(label_feature).unsqueeze(
            1).repeat(1, seq_len, 1, 1)

        temp_features = self.fc_3(torch.tanh(token_features * label_features))
        temp_features = self.fc_a(temp_features)
        scores = torch.sigmoid(temp_features.permute(
            0, 1, 3, 2)).permute(0, 1, 3, 2)
        weighted_label_features = scores * label_features
        fused_features = token_features + weighted_label_features
        # fused_features = token_features *scores
        # fused_features = token_features
        fused_features = torch.tanh(self.fc_5(fused_features))
        # return (fused_features.contiguous().view(batch_size, seq_len, -1),)
        return (fused_features, scores.squeeze(-1))


class Label_Fusion_Layer_for_Seq(nn.Module):
    def __init__(self, hidden_size, label_num, label_emb_size=300):
        super(Label_Fusion_Layer_for_Seq, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_2 = nn.Linear(label_emb_size, self.hidden_size, bias=False)
        self.fc_3 = nn.Linear(self.hidden_size, self.hidden_size)

        # attention, which token is more import for some token
        self.fc_4 = nn.Linear(self.hidden_size, 1)

    def forward(self, token_embs, label_embs):
        # [bs, hidden_size]; [bs, label_num, 300]

        batch_size = token_embs.shape[0]
        token_features = token_embs  # [bs, hidden_dim]
        token_features = self.fc_1(token_features).unsqueeze(1).repeat(
            1, self.label_num, 1)  # [bs, label_num, hidden_dim]

        label_features = self.fc_2(label_embs)  # [bs, label_num, hidden_dim]

        # temp_features = self.fc_3(torch.tanh(
        #     token_features*label_features)).view(-1, self.hidden_size)  # [bs * label_num, hidden_dim]
        # temp_features = self.fc_3(torch.tanh(
        #     token_features*label_features)).contiguous().view(batch_size, -1)  # [bs * label_num, hidden_dim]
        # print(temp_features.shape)
        temp_features = self.fc_3(torch.tanh(
            token_features*label_features))
        return temp_features
        # return (temp_features.view(batch_size, -1),)
        # coefficient = self.fc_4(temp_features)  # [bs * seq_len * label_num, 1]
        # coefficient = self.view(batch_size, seq_len,
        #                         self.label_num).permute(0, 2, 1)  # [bs, label_num, seq_len]


class Label_Attn_Fusion_Layer_for_token(nn.Module):
    def __init__(self, hidden_size, label_num, num_attention_heads, attention_probs_dropout_prob, label_emb_size=300):
        super(Label_Attn_Fusion_Layer_for_token, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.attn_layer = SelfAttention(
            self.hidden_size, num_attention_heads, attention_probs_dropout_prob)

    def forward(self, token_embs, label_embs, token_mask, label_mask):
        # [bs, seq_len, hidden_size]; [label_num, label_seq_len, 300]

        # print(token_embs.shape)
        # print(label_embs.shape)

        batch_size, seq_len = token_embs.shape[:2]

        token_features = token_embs.unsqueeze(
            1).repeat(1,  self.label_num, 1, 1)
        label_features = label_embs.unsqueeze(
            0).repeat(batch_size, 1, 1, 1)

        extend_token_mask = token_mask.unsqueeze(
            1).repeat(1, self.label_num, 1)
        extend_label_mask = label_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        # [bs, label_num, seq_len + label seq_len, hidden_size]
        catted_features = torch.cat((token_features, label_features), dim=2)
        catted_features = catted_features.contiguous().view(
            batch_size * self.label_num, -1, self.hidden_size)

        catted_mask = torch.cat((extend_token_mask, extend_label_mask), dim=2)
        catted_mask = catted_mask.contiguous().view(
            batch_size * self.label_num, -1)
        attn_mask = self.get_extended_attention_mask(
            catted_mask, catted_features.shape)

        fused_features = self.attn_layer(
            catted_features, catted_features, catted_features, attn_mask)[0]

        fused_features = fused_features.contiguous().view(
            batch_size, self.label_num, -1, self.hidden_size).permute(0, 2, 1, 3)
        # print(fused_features.shape)
        return (fused_features,)

    def get_extended_attention_mask(self, attention_mask, input_shape):
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


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(
            hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            : -2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)
        return outputs


class Label_Attn_Fusion_Layer_for_token_backup(nn.Module):
    def __init__(self, hidden_size, label_num, label_emb_size=300):
        super(Label_Attn_Fusion_Layer_for_token, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_2 = nn.Linear(label_emb_size, self.hidden_size, bias=False)
        self.fc_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_a = nn.Linear(self.hidden_size, 1)
        self.fc_fused = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token_embs, label_embs, return_label_embs=False):
        # [bs, seq_len, hidden_size]; [bs, label_num, 300]

        batch_size, seq_len = token_embs.shape[:2]

        token_features = self.fc_1(token_embs)
        # token_features = token_embs
        label_features = self.fc_2(label_embs)

        # cos = nn.CosineSimilarity(dim=-1)

        token_features = token_features.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)
        label_features = label_features.unsqueeze(1).repeat(1, seq_len, 1, 1)

        scores = self.get_scores_by_biliear(token_features, label_features)
        weighted_label_features = scores * label_features
        fused_features = token_features + weighted_label_features
        fused_features = torch.tanh(self.fc_fused(fused_features))
        output = (fused_features.contiguous().view(batch_size, seq_len, -1),)
        if return_label_embs:
            weighted_label_features = torch.mean(weighted_label_features,
                                                 dim=1)
            # weighted_label_features = weighted_label_features.contiguous().view(
            #     batch_size, -1)
            # weighted_label_features = weighted_label_features.contiguous().view(
            #     batch_size, seq_len, -1)
            output += (weighted_label_features,)
        return output

    def get_scores_by_biliear(self, token_features, label_features):
        # token_features = self.fc_1(token_feature).unsqueeze(2).repeat(
        #     1, 1, self.label_num, 1)  # [bs , seq_len, label_num, hidden_dim]

        # label_features = self.fc_2(label_feature).unsqueeze(
        #     1).repeat(1, seq_len, 1, 1)

        temp = self.fc_3(torch.tanh(token_features * label_features))
        # temp = torch.tanh(self.fc_a(temp))
        temp = self.fc_a(temp)
        scores = torch.sigmoid(temp.permute(
            0, 1, 3, 2)).permute(0, 1, 3, 2)
        return scores

    def get_scores_by_cosine_similarity(self, token_features, label_features):
        batch_size, seq_len = token_features.shape[:2]
        cos = nn.CosineSimilarity(dim=-1)
        # token_features_ = token_features.contiguous().view(-1, self.hidden_size)
        # label_features_ = label_features.contiguous().view(-1, self.hidden_size)
        scores = cos(token_features, label_features)  # [-1,1]
        scores = torch.sigmoid(scores).unsqueeze(-1)
        # print(scores.shape)
        return scores


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))

        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(
            input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(
            total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)

    def forward(self, queries, keys, values, src_mask=None):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if src_mask is not None:
            logits = logits.masked_fill(src_mask, -np.inf)

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2],
                                     : logits.shape[-1]].type_as(logits.data)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (
            kernel_size - 1, 0) if pad_type == 'left' else (kernel_size//2, (kernel_size - 1)//2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size,
                              kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)]*(len(layer_config)-2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * \
        np.exp(np.arange(num_timescales).astype(
            np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * \
        np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * \
                (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))
