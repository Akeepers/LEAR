from numpy.core.numeric import outer
import torch
import numpy as np
from utils.common import LabelSpan, LabelSpanWithScore
from collections import namedtuple, Counter
from torch import tensor


class MetricsCalculator4Ner():
    def __init__(self, args, processor) -> None:
        self.model_name = args.model_name
        self.processor = processor
        self.reset()

    def update(self, infer_starts, infer_ends, golden_labels, seq_lens, match_label_ids=None, tokens=None, match_pattern="default", is_logits=True):
        decoded_infers = self.processor.decode_label(infer_starts, infer_ends, seq_lens,
                                                     is_logits=is_logits, batch_match_label_ids=match_label_ids)[0]
        self.goldens.extend(golden_labels)
        self.infers.extend(decoded_infers)
        self.corrects.extend(
            [list(set(seq_infers) & set(seq_goldens)) for seq_infers,
                seq_goldens in zip(decoded_infers, golden_labels)]
        )
        if tokens != None:
            self.tokens.extend(tokens)

    def reset(self):
        self.goldens = []
        self.infers = []
        self.corrects = []
        self.tokens = []

    def get_metrics(self, metrics_type='micro_f1', info_type="default"):
        assert len(self.goldens) == len(self.infers) == len(self.corrects)
        infer_num = sum([len(items) for items in self.infers])
        golden_num = sum([len(items) for items in self.goldens])
        correct_num = sum([len(items) for items in self.corrects])
        precision, recall, f1 = self.calculate_f1(
            golden_num, infer_num, correct_num)
        metrics_info = {}
        metrics_info['general'] = {"precision": precision,
                                   "recall": recall, "f1": f1, "infer_num": infer_num, "golden_num": golden_num, "correct_num": correct_num}
        if info_type == "detail":
            infer_counter = Counter(
                [item.label_id for items in self.infers for item in items])
            golden_counter = Counter(
                [item.label_id for items in self.goldens for item in items])
            correct_counter = Counter(
                [item.label_id for items in self.corrects for item in items])
            for label_id, golden_num in golden_counter.items():
                infer_num = infer_counter.get(label_id, 0)
                correct_num = correct_counter.get(label_id, 0)
                precision, recall, f1 = self.calculate_f1(
                    golden_num, infer_num, correct_num)
                metrics_info[self.processor.id2label[label_id]] = {"precision": precision, "recall": recall,
                                                                   "f1": f1, "infer_num": infer_num, "golden_num": golden_num, "correct_num": correct_num}
        return metrics_info

    def get_results(self, output_diff=True):
        assert len(self.tokens) == len(self.infers)
        return self.processor.build_output_results(self.tokens, self.infers, self.goldens if output_diff else None)

    def calculate_f1(self, label_num, infer_num, correct_num):
        """calcuate f1, precision, recall"""
        if infer_num == 0:
            precision = 0.0
        else:
            precision = correct_num * 1.0 / infer_num
        if label_num == 0:
            recall = 0.0
        else:
            recall = correct_num * 1.0 / label_num
        if correct_num == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1


class MetricsCalculator4ED():
    def __init__(self, args, processor) -> None:
        self.model_name = args.model_name
        self.processor = processor
        self.reset()

    def update(self, infer_starts, infer_ends, golden_labels, seq_lens, match_label_ids=None, tokens=None, ids=None, match_pattern="default", is_logits=True):
        decoded_infers = self.processor.decode_label(infer_starts, infer_ends, seq_lens,
                                                     is_logits=is_logits, batch_match_label_ids=match_label_ids)[0]
        if match_pattern == "maven":
            temp = []
            for seq_infers, seq_goldens in zip(decoded_infers, golden_labels):
                golden_label_dict = {(label.start_idx, label.end_idx): label.label_id for label in seq_goldens}
                cur_infers = []
                for infer in seq_infers:
                    if (infer.start_idx, infer.end_idx) in golden_label_dict:
                        cur_infers.append(infer)
                temp.append(cur_infers)
            decoded_infers = temp
        self.goldens.extend(golden_labels)
        self.infers.extend(decoded_infers)
        self.corrects.extend(
            [list(set(seq_infers) & set(seq_goldens)) for seq_infers,
                seq_goldens in zip(decoded_infers, golden_labels)]
        )
        if tokens != None:
            self.tokens.extend(tokens)
        if ids != None:
            self.ids.extend(ids)

    def reset(self):
        self.goldens = []
        self.infers = []
        self.corrects = []
        self.tokens = []
        self.ids = []

    def get_metrics(self, metrics_type='micro_f1', info_type="default"):
        assert len(self.goldens) == len(self.infers) == len(self.corrects)
        infer_num = sum([len(items) for items in self.infers])
        golden_num = sum([len(items) for items in self.goldens])
        correct_num = sum([len(items) for items in self.corrects])
        precision, recall, f1 = self.calculate_f1(golden_num, infer_num, correct_num)
        metrics_info = {}
        metrics_info['general'] = {"precision": precision,
                                   "recall": recall, "f1": f1, "infer_num": infer_num, "golden_num": golden_num, "correct_num": correct_num}
        if info_type == "detail":
            infer_counter = Counter(
                [item.label_id for items in self.infers for item in items])
            golden_counter = Counter(
                [item.label_id for items in self.goldens for item in items])
            correct_counter = Counter(
                [item.label_id for items in self.corrects for item in items])
            for label_id, golden_num in golden_counter.items():
                infer_num = infer_counter.get(label_id, 0)
                correct_num = correct_counter.get(label_id, 0)
                precision, recall, f1 = self.calculate_f1(
                    golden_num, infer_num, correct_num)
                metrics_info[self.processor.id2label[label_id]] = {"precision": precision, "recall": recall,
                                                                   "f1": f1, "infer_num": infer_num, "golden_num": golden_num, "correct_num": correct_num}
        return metrics_info

    def get_results(self, output_diff=True, append_ids=False):
        assert len(self.tokens) == len(self.infers)
        return self.processor.build_output_results(self.tokens, self.infers, self.goldens if output_diff else None, ids=(self.ids if append_ids else None))

    def calculate_f1(self, label_num, infer_num, correct_num):
        """calcuate f1, precision, recall"""
        if infer_num == 0:
            precision = 0.0
        else:
            precision = correct_num * 1.0 / infer_num
        if label_num == 0:
            recall = 0.0
        else:
            recall = correct_num * 1.0 / label_num
        if correct_num == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1


def get_span_crf(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        # output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    output = [LabelSpan(start_idx=int(item[1]), end_idx=int(
        item[2]), label_id=item[0]) for item in chunks]
    # return chunks
    return output


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred > thresh) == y_true.bool()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred, y_true, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9, sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()


def extract_span(starts, ends, position_dict=None, limit=0.5):
    assert len(starts.shape) == 3 and len(ends.shape) == 3
    label_num = starts.shape[-1]
    # start_idxes, end_idxes = [], []
    span_list = []
    for label_idx in range(label_num):
        cur_start_idxes = np.where(starts[:, :label_idx] > limit)
        cur_end_idxes = np.where(ends[:, :label_idx] > limit)
        if position_dict is not None:
            cur_start_idxes = [position_dict[idx] for idx in cur_start_idxes]
            cur_end_idxes = [position_dict[idx] for idx in cur_end_idxes]
        # start_idxes.append(cur_start_idxes)
        # end_idxes.append(cur_end_idxes)
        cur_spans = []
        for start_idx in cur_start_idxes:
            ends = cur_end_idxes[cur_end_idxes >= start_idx]
            if len(ends) > 0:
                cur_spans.append(LabelSpan(start_idx=start_idx,
                                           end_idx=ends[0], label_id=label_idx))
        cur_spans = list(set(cur_spans))
        span_list.append(cur_spans)
    return span_list


def restore_subtoken(tokens, return_tokens=False):
    """tokens include [cls], [sep]"""
    position_dict, original_tokens = {}, []
    start_dict, end_dict = {}, {}
    for i, subtoken in enumerate(tokens):
        if subtoken.startswith("##"):
            assert len(original_tokens) != 0
            original_tokens[-1] += subtoken[2:]
            position_dict[i] = len(original_tokens) - 1
        else:
            position_dict[i] = len(original_tokens)

            start_dict[len(original_tokens)] = i
            if i != 0:
                end_dict[len(original_tokens) - 1] = i-1

            original_tokens.append(subtoken)

    end_dict[len(original_tokens) - 1] = i-1
    if return_tokens:
        return position_dict, original_tokens, start_dict, end_dict
    return position_dict, start_dict, end_dict


def count_span(starts, ends, input_mask, match_logits, match_labels, return_span=False, s_limit=0.5, e_limit=0.5):
    """ for nested"""
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)

    # [batch_size, seq_len]
    batch_size, seq_len, class_num = starts.size()

    # [batch_size, seq_len, class_num]
    extend_input_mask = input_mask.unsqueeze(
        -1).expand(-1, -1, class_num)

    start_label_mask = extend_input_mask.unsqueeze(
        -2).expand(-1, -1, seq_len, -1).bool()
    end_label_mask = extend_input_mask.unsqueeze(
        -3).expand(-1, seq_len, -1, -1).bool()

    match_labels = match_labels.bool()

    # [batch_size, seq_len, seq_len, class_num]
    match_infer = match_logits > 0
    # match_infer = match_labels.bool()
    # [batch_size, seq_len, class_num]
    start_infer = starts > 0
    # [batch_size, seq_len, class_num]
    end_infer = ends > 0
    start_infer = start_infer.bool()
    end_infer = end_infer.bool()

    # match_infer = torch.ones_like(match_infer)

    match_infer = (
        match_infer & start_infer.unsqueeze(2).expand(-1, -1, seq_len, -1)
        & end_infer.unsqueeze(1).expand(-1, seq_len, -1, -1))

    # match_label_mask = (start_label_mask & end_label_mask)
    # match_label_mask = torch.triu(match_label_mask, 0)
    match_label_mask = torch.triu((start_label_mask & end_label_mask).permute(
        0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
        batch_size, class_num, seq_len, seq_len).permute(0, 2, 3, 1)

    # [batch_size, seq_len, seq_len, class_num]
    match_infer = match_infer & match_label_mask

    match_correct = (match_infer & match_labels)
    match_infer_cnt = match_infer.long().sum()
    match_correct_cnt = match_correct.long().sum()
    match_golden_cnt = match_labels.long().sum()

    output = (match_infer_cnt, match_correct_cnt, match_golden_cnt)
    if return_span:
        span_list = [[] for batch_idx in range(batch_size)]
        golden_span_list = [[] for batch_idx in range(batch_size)]
        items = torch.where(match_infer == True)
        if len(items[0]) != 0:
            for idx in range(len(items[0])):
                batch_idx = int(items[0][idx])
                start_idx = int(items[1][idx])
                end_idx = int(items[2][idx])
                label_id = int(items[3][idx])
                span_list[batch_idx].append(
                    LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))
        assert sum([len(item) for item in span_list]) == match_infer_cnt

        items = torch.where(match_labels == True)
        if len(items[0]) != 0:
            for idx in range(len(items[0])):
                batch_idx = int(items[0][idx])
                start_idx = int(items[1][idx])
                end_idx = int(items[2][idx])
                label_id = int(items[3][idx])
                golden_span_list[batch_idx].append(
                    LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))
        assert sum([len(item)
                    for item in golden_span_list]) == match_golden_cnt

        output += (span_list, golden_span_list)
    return output


def extract_span_v1(starts, ends, seqlens=None, position_dict=None, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
    """ for nested"""
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)
    if seqlens is not None:
        assert starts.shape[0] == seqlens.shape[0]

    label_num = starts.shape[-1]
    span_list = [[] for _ in range(starts.shape[0])]
    for batch_idx in range(starts.shape[0]):
        for label_idx in range(label_num):
            if seqlens is not None:
                cur_start_idxes = np.where(
                    starts[batch_idx, :seqlens[batch_idx], label_idx] > s_limit)
                cur_end_idxes = np.where(
                    ends[batch_idx, :seqlens[batch_idx], label_idx] > e_limit)
            else:
                cur_start_idxes = np.where(
                    starts[batch_idx, :, label_idx] > s_limit)
                cur_end_idxes = np.where(
                    ends[batch_idx, :, label_idx] > e_limit)

            if cur_start_idxes[0].size == 0 or cur_end_idxes[0].size == 0:
                continue
            # cur_start_idxes = np.array([pos for pos in cur_start_idxes])
            # cur_end_idxes = np.array([pos[0] for pos in cur_end_idxes])
            cur_start_idxes = cur_start_idxes[0]
            cur_end_idxes = cur_end_idxes[0]
            # print(cur_start_idxes)
            # print(cur_end_idxes)
            if position_dict is not None:
                cur_start_idxes = np.unique(np.array([position_dict[batch_idx][idx]
                                                      for idx in cur_start_idxes]))
                cur_end_idxes = np.unique(np.array([position_dict[batch_idx][idx]
                                                    for idx in cur_end_idxes]))
            cur_spans = []
            # print(cur_start_idxes)
            # print(cur_end_idxes)
            for start_idx in cur_start_idxes:
                cur_ends = cur_end_idxes[cur_end_idxes >= start_idx]
                if len(cur_ends) > 0:
                    cur_spans.append(LabelSpan(start_idx=start_idx,
                                               end_idx=cur_ends[0], label_id=int(label_idx)))
            cur_spans = list(set(cur_spans))
            span_list[batch_idx].extend(cur_spans)
    return (span_list,)


def extract_span_v2(starts, ends, seqlens=None, position_dict=None, s_limit=0.5, e_limit=0.5):
    """ for nested"""
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)
    if seqlens is not None:
        assert starts.shape[0] == seqlens.shape[0]

    label_num = starts.shape[-1]
    span_list = [[] for _ in range(starts.shape[0])]
    for batch_idx in range(starts.shape[0]):
        for label_idx in range(label_num):
            if seqlens is not None:
                cur_start_idxes = np.where(
                    starts[batch_idx, :seqlens[batch_idx], label_idx] > s_limit)
                cur_end_idxes = np.where(
                    ends[batch_idx, :seqlens[batch_idx], label_idx] > e_limit)
            else:
                cur_start_idxes = np.where(
                    starts[batch_idx, :, label_idx] > s_limit)
                cur_end_idxes = np.where(
                    ends[batch_idx, :, label_idx] > e_limit)

            if cur_start_idxes[0].size == 0 or cur_end_idxes[0].size == 0:
                continue
            cur_start_idxes = cur_start_idxes[0]
            cur_end_idxes = cur_end_idxes[0]
            if position_dict is not None:
                cur_start_idxes = np.unique(np.array([position_dict[batch_idx][idx]
                                                      for idx in cur_start_idxes]))
                cur_end_idxes = np.unique(np.array([position_dict[batch_idx][idx]
                                                    for idx in cur_end_idxes]))
            cur_spans = []

            for start_idx in cur_start_idxes:
                cur_ends = cur_end_idxes[cur_end_idxes >= start_idx]
                if len(cur_ends) > 0:
                    cur_spans.append(LabelSpan(start_idx=start_idx,
                                               end_idx=cur_ends[0], label_id=int(label_idx)))
            for end_idx in cur_end_idxes:
                cur_starts = cur_start_idxes[cur_start_idxes <= end_idx]
                if len(cur_starts) > 0:
                    cur_spans.append(LabelSpan(start_idx=cur_starts[-1],
                                               end_idx=end_idx, label_id=int(label_idx)))
            cur_spans = list(set(cur_spans))
            span_list[batch_idx].extend(cur_spans)
    return span_list


def extract_span_v3(starts, ends, seqlens=None, position_dict=None, scores=None, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)
    if return_span_score:
        assert scores is not None
        span_score_list = [[] for _ in range(starts.shape[0])]
    if seqlens is not None:
        assert starts.shape[0] == seqlens.shape[0]
    if return_cnt:
        span_cnt = 0
    label_num = starts.shape[-1]
    span_list = [[] for _ in range(starts.shape[0])]

    for batch_idx in range(starts.shape[0]):
        for label_idx in range(label_num):

            cur_spans = []

            seq_start_labels = starts[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                               ] if seqlens is not None else starts[batch_idx, :, label_idx]
            seq_end_labels = ends[batch_idx, :, label_idx][: seqlens[batch_idx]
                                                           ] if seqlens is not None else ends[batch_idx, :, label_idx]

            # if np.all(seq_start_labels <= s_limit) or np.all(seq_end_labels <= e_limit):  # no span
            #     continue
            start_prob, start_idx = -1, -1
            for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                if token_start_prob > s_limit and token_start_prob > start_prob:  # new start
                    start_prob = token_start_prob
                    start_idx = token_idx
                if token_end_prob > e_limit and start_prob > s_limit:  # end
                    if return_span_score:
                        start_score = float(
                            scores[batch_idx][start_idx][label_idx])
                        end_score = float(
                            scores[batch_idx][token_idx][label_idx])
                    if position_dict is not None:
                        start_idx = position_dict[batch_idx][start_idx]
                        end_idx = position_dict[batch_idx][token_idx]
                    else:
                        end_idx = token_idx
                    assert start_idx <= end_idx
                    if return_span_score:
                        cur_spans.append(LabelSpanWithScore(start_idx=start_idx,
                                                            end_idx=end_idx, label_id=label_idx, start_score=start_score, end_score=end_score))
                    else:
                        cur_spans.append(LabelSpan(start_idx=start_idx,
                                                   end_idx=end_idx, label_id=label_idx))
                    start_prob, start_idx = -1, -1

            cur_spans = list(set(cur_spans))
            if return_cnt:
                span_cnt += len(cur_spans)
            if return_span_score:
                span_score_list[batch_idx].extend(
                    [(item.start_score, item.end_score) for item in cur_spans])
                span_list[batch_idx].extend([LabelSpan(
                    start_idx=item.start_idx, end_idx=item.end_idx, label_id=item.label_id) for item in cur_spans])
            else:
                span_list[batch_idx].extend(cur_spans)
    output = (span_list,)
    if return_cnt:
        output += (span_cnt,)
    if return_span_score:
        output += (span_score_list,)
    return output


def extract_span_relation(starts, ends, seqlens=None, position_dict=None, s_limit=0.5, e_limit=0.5, return_cnt=False):
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)
    if seqlens is not None:
        assert starts.shape[0] == seqlens.shape[0]
    if return_cnt:
        span_cnt = 0
    label_num = starts.shape[-1]
    span_list = [[] for _ in range(starts.shape[0])]

    for batch_idx in range(starts.shape[0]):
        for label_idx in range(label_num):

            cur_spans = []

            seq_start_labels = starts[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                               ] if seqlens is not None else starts[batch_idx, :, label_idx]
            seq_end_labels = ends[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                           ] if seqlens is not None else ends[batch_idx, :, label_idx]

            # if np.all(seq_start_labels <= s_limit) or np.all(seq_end_labels <= e_limit):  # no span
            #     continue
            start_prob, start_idx = -1, -1
            for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                if token_start_prob > s_limit and token_start_prob > start_prob:  # new start
                    start_prob = token_start_prob
                    start_idx = token_idx
                if token_end_prob > e_limit and start_prob > s_limit:  # end
                    if position_dict is not None:
                        start_idx = position_dict[batch_idx][start_idx]
                        end_idx = position_dict[batch_idx][token_idx]
                    else:
                        end_idx = token_idx
                    assert start_idx <= end_idx
                    cur_spans.append(LabelSpan(start_idx=start_idx,
                                               end_idx=end_idx, label_id=label_idx))
                    start_prob, start_idx = -1, -1

            cur_spans = list(set(cur_spans))
            if return_cnt:
                span_cnt += len(cur_spans)
            span_list[batch_idx].extend(cur_spans)
    if return_cnt:
        return span_list, span_cnt
    return span_list


def extract_span_v5(starts, ends, seqlens=None, position_dict=None, scores=None, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)
    if return_span_score:
        assert scores is not None
        span_score_list = [[] for _ in range(starts.shape[0])]
    if seqlens is not None:
        assert starts.shape[0] == seqlens.shape[0]
    if return_cnt:
        span_cnt = 0
    label_num = starts.shape[-1]
    span_list = [[] for _ in range(starts.shape[0])]

    for batch_idx in range(starts.shape[0]):
        for label_idx in range(label_num):

            cur_spans = []

            seq_start_labels = starts[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                               ] if seqlens is not None else starts[batch_idx, :, label_idx]
            seq_end_labels = ends[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                           ] if seqlens is not None else ends[batch_idx, :, label_idx]

            # if np.all(seq_start_labels <= s_limit) or np.all(seq_end_labels <= e_limit):  # no span
            #     continue
            start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1
            for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                if token_start_prob > s_limit:
                    if end_idx != -1:  # build span
                        if return_span_score:
                            cur_spans.append(LabelSpanWithScore(start_idx=start_idx,
                                                                end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                        else:
                            cur_spans.append(LabelSpan(start_idx=start_idx,
                                                       end_idx=end_idx, label_id=label_idx))
                        start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1  # reset state
                    if token_start_prob > start_prob:  # new start, if pre prob is lower, drop it
                        start_prob = token_start_prob
                        start_idx = token_idx
                if token_end_prob > e_limit and start_prob > s_limit:  # end
                    if token_end_prob > end_prob:
                        end_prob = token_end_prob
                        end_idx = token_idx
            if end_idx != -1:
                if return_span_score:
                    cur_spans.append(LabelSpanWithScore(start_idx=start_idx,
                                                        end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                else:
                    cur_spans.append(LabelSpan(start_idx=start_idx,
                                               end_idx=end_idx, label_id=label_idx))
            cur_spans = list(set(cur_spans))
            if return_cnt:
                span_cnt += len(cur_spans)
            if return_span_score:
                span_score_list[batch_idx].extend(
                    [(item.start_score, item.end_score) for item in cur_spans])
                span_list[batch_idx].extend([LabelSpan(
                    start_idx=item.start_idx, end_idx=item.end_idx, label_id=item.label_id) for item in cur_spans])
            else:
                span_list[batch_idx].extend(cur_spans)
    output = (span_list,)
    if return_cnt:
        output += (span_cnt,)
    if return_span_score:
        output += (span_score_list,)
    return output


def extract_span_v4(starts, ends, seqlens=None, position_dict=None, scores=None, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
    assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
        starts.shape, ends.shape)
    if return_span_score:
        assert scores is not None
    if seqlens is not None:
        assert starts.shape[0] == seqlens.shape[0]
    if return_cnt:
        span_cnt = 0
    label_num = starts.shape[-1]
    span_list = [[] for _ in range(starts.shape[0])]

    for batch_idx in range(starts.shape[0]):
        for label_idx in range(label_num):

            cur_spans = []

            seq_start_labels = starts[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                               ] if seqlens is not None else starts[batch_idx, :, label_idx]
            seq_end_labels = ends[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                           ] if seqlens is not None else ends[batch_idx, :, label_idx]

            # if np.all(seq_start_labels <= s_limit) or np.all(seq_end_labels <= e_limit):  # no span
            #     continue
            start_prob, start_idx = -1, -1
            for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                if token_start_prob > s_limit and token_start_prob > start_prob:  # new start
                    start_prob = token_start_prob
                    start_idx = token_idx
                if token_end_prob > e_limit and start_prob > s_limit:  # end
                    if position_dict is not None:
                        start_idx = position_dict[batch_idx][start_idx]
                        end_idx = position_dict[batch_idx][token_idx]
                    else:
                        end_idx = token_idx
                    assert start_idx <= end_idx
                    cur_spans.append(LabelSpan(start_idx=start_idx,
                                               end_idx=end_idx, label_id=label_idx))
                    start_prob, start_idx = -1, -1

            end_prob, end_idx, token_idx = -1, -1, len(seq_end_labels) - 1
            while token_idx >= 0:
                token_end_prob = seq_end_labels[token_idx]
                token_start_prob = seq_start_labels[token_idx]
                if token_end_prob > e_limit and token_end_prob > end_prob:  # new end
                    end_prob = token_end_prob
                    end_idx = token_idx
                if token_start_prob > s_limit and end_prob > e_limit:  # start
                    if position_dict is not None:
                        end_idx = position_dict[batch_idx][end_idx]
                        start_idx = position_dict[batch_idx][token_idx]
                    else:
                        start_idx = token_idx
                    assert start_idx <= end_idx
                    cur_spans.append(LabelSpan(start_idx=start_idx,
                                               end_idx=end_idx, label_id=label_idx))
                    end_prob, end_idx = -1, -1
                token_idx -= 1

            cur_spans = list(set(cur_spans))
            if return_cnt:
                span_cnt += len(cur_spans)
            span_list[batch_idx].extend(cur_spans)
    if return_cnt:
        return span_list, span_cnt
    return span_list


def calculate_f1(label_num, infer_num, correct_num):
    """calcuate f1, precision, recall"""
    if infer_num == 0:
        precision = 0.0
    else:
        precision = correct_num * 1.0 / infer_num
    if label_num == 0:
        recall = 0.0
    else:
        recall = correct_num * 1.0 / label_num
    if correct_num == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def build_trigger_list_from_golden(golden_labels, return_cnt=True):
    trigger_list, cnt = [], 0
    for events in golden_labels:
        cur_triggers = []
        for event in events:
            cur_triggers.append(
                (event['trigger']['text'], event['trigger']['label']))
        if return_cnt:
            cnt += len(cur_triggers)
        trigger_list.append(cur_triggers)
    if return_cnt:
        return trigger_list, cnt
    return trigger_list


def build_argument_list_from_golden(golden_labels,  return_cnt=True):
    argument_list, cnt = [], 0
    for events in golden_labels:
        cur_arguments = []
        for event in events:
            for argument in event['arguments']:
                cur_arguments.append(
                    (event['trigger']['text'], event['trigger']['label'], argument['text'], argument['label']))
        if return_cnt:
            cnt += len(cur_arguments)
        argument_list.append(cur_arguments)
    if return_cnt:
        return argument_list, cnt
    return argument_list


def count_arguments(golden_labels):
    cnt = 0
    for events in golden_labels:
        for event in events:
            cnt += len(event['arguments'])
    return cnt


def count_relation(golden_labels):
    cnt = 0
    for relations in golden_labels:
        cnt += len(relations)
    return cnt


def build_entity_list_from_golden(golden_labels,  return_cnt=True):
    label_list, cnt = [], 0
    for labels in golden_labels:
        cur_labels = []
        for label in labels:
            cur_labels.append((label['text'], label['label']))
        if return_cnt:
            cnt += len(cur_labels)
        label_list.append(cur_labels)
    if return_cnt:
        return label_list, cnt
    return label_list


def build_event_v2(tokens, trigger, arguments, id2trigger, id2argument, joiner_symbol=" "):
    argument_list = [{'text': joiner_symbol.join(tokens[argument.start_idx:argument.end_idx+1]),
                      'label': id2argument[argument.label_id]}
                     for argument in arguments]

    event = {
        'trigger': joiner_symbol.join(tokens[trigger.start_idx:trigger.end_idx+1]),
        'event_type': id2trigger[trigger.label_id],
        'argument_list': argument_list
    }
    return event


def build_relation_list(tokens, subject, objects, id2relation, joiner_symbol=" "):
    triple_list = [{'subject': joiner_symbol.join(tokens[subject.start_idx:subject.end_idx+1]),
                    'relation': id2relation[obj.label_id],
                    'object': joiner_symbol.join(tokens[obj.start_idx:obj.end_idx+1])}
                   for obj in objects]
    return triple_list


def build_events(tokens, trigger, labels, id2trigger, id2argument, joiner_symbol=" ", event_cnt=-1):
    argument_list = []
    # print(tokens)
    # print(type(tokens))
    for label in labels:
        argument_list.append({'text': joiner_symbol.join(
            tokens[label.start_idx:label.end_idx+1]), 'label': id2argument[label.label_id]})
    return {
        'trigger': joiner_symbol.join(tokens[trigger.start_idx:trigger.end_idx+1]),
        'event_type': id2trigger[trigger.label_id],
        'argument_list': argument_list
    }


def extend_tensor(tensor, extend_mapping):
    return tensor[extend_mapping]


def build_evaluate_second_label_ids(batch_input_data, batch_infer_data, second_label_dict, label_num):
    assert len(batch_infer_data) == batch_golden_labels
    second_label_start_list, second_label_end_list = [], []
    seqlen = batch_input_data['token_ids'].shape[1]
    for seq_infers, example_id in zip(batch_infer_data, batch_input_data['ids']):
        for infer_label in seq_infers:
            seq_second_label_start_ids = np.zeros(seqlen, label_num)
            seq_second_label_end_ids = np.zeros(seqlen, label_num)
            key = (example_id, infer_label.start_idx,
                   infer_label.end_idx, infer_label.label_id)
            if key in second_label_dict:
                for second_label in second_label_dict[key]:
                    seq_second_label_start_ids[second_label.start_idx][second_label.label_id] = 1
                    seq_second_label_end_ids[second_label.end_idx][second_label.label_id] = 1
            second_label_start_list.append(seq_second_label_start_ids)
            second_label_end_list.append(seq_second_label_end_ids)
    second_label_start_ids = torch.tensor(
        second_label_start_list, dtype=torch.float32).to(batch_input_data['token_ids'].device)
    second_label_end_ids = torch.tensor(
        second_label_end_list, dtype=torch.float32).to(batch_input_data['token_ids'].device)
    return second_label_start_ids, second_label_end_ids


def build_second_label_for_evaluation(batch_input_data, batch_first_label_infer_data, second_label_dict, flatten_trigger_span_list=None, batch_position_dict=None):
    second_label_list = []
    idx = -1
    for batch_idx, (seq_infers, example_id) in enumerate(zip(batch_first_label_infer_data, batch_input_data['ids'])):
        for infer_label in seq_infers:
            idx += 1
            assert infer_label == flatten_trigger_span_list[idx], " {},{}".format(
                infer_label, flatten_trigger_span_list[idx])
            key = (example_id, int(infer_label.start_idx),
                   int(infer_label.end_idx), int(infer_label.label_id))
            # print(key)
            if key in second_label_dict:
                if batch_position_dict is not None:
                    second_label_list.append([LabelSpan(start_idx=batch_position_dict[batch_idx][second_label.start_idx], end_idx=batch_position_dict[batch_idx]
                                                        [second_label.end_idx], label_id=second_label.label_id) for second_label in second_label_dict[key]])
                else:
                    second_label_list.append(second_label_dict[key])
            else:
                second_label_list.append([])
    return second_label_list

# def extract_span_v2(starts, ends, seqlens=None, position_dict=None, s_limit=0.5, e_limit=0.5, return_cnt=False):
#     """for overlap"""
#     assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
#         starts.shape, ends.shape)
#     if seqlens is not None:
#         assert starts.shape[0] == seqlens.shape[0]
#     if return_cnt:
#         span_cnt = 0

#     def get_span(start_ids, end_ids):
#         """
#         every id can only be used once
#         get span set from position start and end list
#         input: [1, 2, 10] [4, 12]
#         output: set((2, 4), (10, 12))
#         """
#         start_ids = sorted(start_ids)
#         end_ids = sorted(end_ids)
#         start_pointer = 0
#         end_pointer = 0
#         len_start = len(start_ids)
#         len_end = len(end_ids)
#         couple_dict = {}
#         while start_pointer < len_start and end_pointer < len_end:
#             if start_ids[start_pointer] == end_ids[end_pointer]:
#                 couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
#                 start_pointer += 1
#                 end_pointer += 1
#                 continue
#             if start_ids[start_pointer] < end_ids[end_pointer]:
#                 couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
#                 start_pointer += 1
#                 continue
#             if start_ids[start_pointer] > end_ids[end_pointer]:
#                 end_pointer += 1
#                 continue
#         result = [(couple_dict[end], end) for end in couple_dict]
#         result = set(result)
#         return list(result)

#     label_num = starts.shape[-1]
#     span_list = [[] for _ in range(starts.shape[0])]
#     for batch_idx in range(starts.shape[0]):
#         for label_idx in range(label_num):
#             if seqlens is not None:
#                 cur_start_idxes = np.where(
#                     starts[batch_idx, :seqlens[batch_idx], label_idx] > s_limit)
#                 cur_end_idxes = np.where(
#                     ends[batch_idx, :seqlens[batch_idx], label_idx] > e_limit)
#             else:
#                 cur_start_idxes = np.where(
#                     starts[batch_idx, :, label_idx] > s_limit)
#                 cur_end_idxes = np.where(
#                     ends[batch_idx, :, label_idx] > e_limit)

#             if cur_start_idxes[0].size == 0 or cur_end_idxes[0].size == 0:
#                 continue
#             # cur_start_idxes = np.array([pos for pos in cur_start_idxes])
#             # cur_end_idxes = np.array([pos[0] for pos in cur_end_idxes])
#             cur_start_idxes = list(cur_start_idxes[0])
#             cur_end_idxes = list(cur_end_idxes[0])

#             if position_dict is not None:
#                 cur_start_idxes = [position_dict[batch_idx][idx]
#                                    for idx in cur_start_idxes]
#                 cur_end_idxes = [position_dict[batch_idx][idx]
#                                  for idx in cur_end_idxes]
#             # print(cur_end_idxes)
#             seq_span = get_span(cur_start_idxes, cur_end_idxes)
#             cur_span = [LabelSpan(start_idx=int(span[0]), end_idx=int(
#                 span[1]), label_id=int(label_idx)) for span in seq_span]
#             if return_cnt:
#                 span_cnt += len(cur_span)
#             span_list[batch_idx].extend(cur_span)

#     def get_bool_ids_greater_than(probs, limit):
#         """
#         get idx of the last dim in prob arraies, which is greater than a limitation
#         input: [[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]
#             0.4
#         output: [[3], [0, 1]]
#         """
#         probs = np.array(probs)
#         dim_len = len(probs.shape)
#         if dim_len > 1:
#             result = []
#             for p in probs:
#                 result.append(get_bool_ids_greater_than(p, limit))
#             return result
#         else:
#             result = []
#             for i, p in enumerate(probs):
#                 if p > limit:
#                     result.append(i)
#             return result

#     if return_cnt:
#         return span_list, span_cnt
#     return span_list

    # starts_idxes = get_bool_ids_greater_than(starts, s_limit)
    # ends_idxes = get_bool_ids_greater_than(ends, e_limit)
    # return get_span(starts_idxes, ends_idxes)


extract_span_func = {
    'v1': extract_span_v1,
    'v2': extract_span_v2,
    'v3': extract_span_v3,
    'v4': extract_span_v4,
    'v5': extract_span_v5
}
