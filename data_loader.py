from torch.utils.data import DataLoader, Dataset
import csv
import json
import copy
import torch
import random
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choice
import utils.common as common
from utils.tokenization import basic_tokenize, convert_to_unicode
from multiprocessing import Pool, cpu_count, Manager


class DataProcessor(object):
    """Base class for data converters for token classification data sets."""

    def get_train_examples(self, input_file):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, input_file):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines


class RelationDataProcessor(DataProcessor):
    """Processor for the relation extraction data set."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if 'id' not in line:
                line['id'] = guid
            examples.append(line)
        return examples

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_len, first_label_map_file, second_label_map_file, data_type="train", pad_id=0, do_lower_case=False):
        # first_label2id = self.get_labels(label_file=first_label_map_file)[1]
        second_label2id = self.get_labels(label_file=second_label_map_file)[1]

        relativate_pos_dict = {}
        for idx in range(-max_seq_len+1, max_seq_len):
            relativate_pos_dict[idx] = len(relativate_pos_dict)

        features = []

        stat_info = {"relation_cnt": 0}
        if data_type != "train":
            object_dict = {}

        for example in examples:
            text = example['text']
            tokens, ori_tokens = [], []
            all_tokens_with_orig_offset = []
            offset = 0
            for token in basic_tokenize(text):
                sub_offset = 0
                if token != ' ':  # skip space
                    ori_tokens.append(token)
                token_lower = token.lower() if do_lower_case else token
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                    proc_sub_token = sub_token
                    if sub_token.startswith('##'):
                        proc_sub_token = sub_token[2:]
                    new_sub_offset = token_lower.find(
                        proc_sub_token, sub_offset)
                    if new_sub_offset > -1:
                        sub_offset = new_sub_offset
                    all_tokens_with_orig_offset.append(
                        (sub_token, offset + sub_offset, token, offset))
                    if new_sub_offset > -1:
                        sub_offset += len(sub_token)
                    else:
                        sub_offset += 1
                offset += len(token)

            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[: max_seq_len - special_tokens_count]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            ori_tokens = [tokenizer.cls_token] + \
                ori_tokens + [tokenizer.sep_token]

            all_tokens_with_orig_offset = [
                ('CLS', 0, '', 0)] + all_tokens_with_orig_offset

            offset_dict = {}
            for i, (_, _, orig_token, offset) in enumerate(all_tokens_with_orig_offset):
                if i == 0:
                    continue
                offset_dict.setdefault(offset, i)
                offset_dict.setdefault(offset + len(orig_token), i + 1)

            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            seq_len = len(tokens_ids)
            token_type_ids = [0] * seq_len
            input_mask = [1] * seq_len
            extra_mask = [0] * seq_len

            padding_length = max_seq_len - len(tokens_ids)
            tokens_ids += [pad_id] * padding_length
            token_type_ids += [pad_id] * padding_length
            input_mask += [pad_id] * padding_length

            extra_mask += [1] * padding_length

            # encoded_results = tokenizer.encode(
            #     example['text'], add_special_tokens=True)
            # tokens = encoded_results.tokens
            # token_ids = encoded_results.ids
            # token_type_ids = encoded_results.type_ids
            # input_mask = encoded_results.attention_mask

            # offset_dict = {}
            # for token_idx in range(len(tokens)-1):  # skip [sep]
            #     if token_idx == 0:  # skip [cls]
            #         continue
            #     token_start, token_end = encoded_results.offsets[token_idx]
            #     offset_dict[token_start] = token_idx
            #     offset_dict[token_end] = token_idx

            # token_ids = token_ids[:max_seq_len]
            # token_type_ids = token_type_ids[:max_seq_len]
            # input_mask = input_mask[:max_seq_len]

            # sep_id = tokenizer.token_to_id('[SEP]')
            # if token_ids[-1] != sep_id:
            #     assert len(token_ids) == max_seq_len
            #     token_ids[-1] = sep_id
            # seq_len = len(token_ids)

            # first_label_start_ids, first_label_end_ids = torch.zeros(
            #     seq_len, dtype=torch.float32), torch.zeros(seq_len, dtype=torch.float32)
            # second_label_start_ids = torch.zeros(
            #     (seq_len, len(second_label2id)), dtype=torch.float32)
            # second_label_end_ids = torch.zeros(
            #     (seq_len, len(second_label2id)), dtype=torch.float32)
            # first_label_start, first_label_end = torch.zeros(
            #     seq_len, dtype=torch.float32), torch.zeros(seq_len, dtype=torch.float32)

            first_label_start_ids, first_label_end_ids = np.zeros(
                (max_seq_len)), np.zeros((max_seq_len))
            second_label_start_ids = np.zeros(
                (max_seq_len, len(second_label2id)))
            second_label_end_ids = np.zeros(
                (max_seq_len, len(second_label2id)))
            first_label_start, first_label_end = np.zeros(
                max_seq_len), np.zeros(max_seq_len)

            relativate_pos_label_ids = np.zeros(max_seq_len)
            second_label_mask = 1.0

            if data_type == "train":
                for first_label in example["subject_list"]:
                    first_ori_start_offset = first_label["segments"]["start_offset"]
                    first_ori_end_offset = first_label["segments"]["end_offset"]
                    try:
                        first_start_idx = offset_dict[first_ori_start_offset]
                        first_end_idx = offset_dict[first_ori_end_offset] - 1
                    except:
                        # logger.warn(tokens)
                        # logger.warn("{},{},{}".format(
                        #     text[first_ori_start_offset:first_ori_end_offset+1], first_ori_start_offset, first_ori_end_offset))
                        errmsg = "subject '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            first_label['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if first_end_idx >= seq_len:
                        errmsg = "subject '{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            first_label['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    first_label_start_ids[first_start_idx] = 1
                    first_label_end_ids[first_end_idx] = 1

                if len(example["subject"]) == 0:
                    second_label_mask = 0.0
                    assert len(example["subject_list"]) == 0
                else:
                    subject_ori_start_offset = example["subject"]["segments"]["start_offset"]
                    subject_ori_end_offset = example["subject"]["segments"]["end_offset"]

                    try:
                        subject_start_idx = offset_dict[subject_ori_start_offset]
                        subject_end_idx = offset_dict[subject_ori_end_offset] - 1
                    except:
                        errmsg = "subject '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            example["subject"]["text"], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if subject_end_idx >= seq_len:
                        # logger.info(tokens[subject_end_idx])
                        errmsg = "subject {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            subject_end_idx, example["subject"]['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    # relativate_pos_label_ids = np.array(
                    #     [relativate_pos_dict[-idx] for idx in range(subject_start_idx, 0, -1)] + [0] * (subject_end_idx-subject_start_idx+1) + [
                    #         relativate_pos_dict[idx] for idx in range(1, max_seq_len - subject_end_idx)
                    #     ]
                    # )

                    first_label_start[subject_start_idx] = 1
                    first_label_end[subject_end_idx] = 1

                    for obj in example["object_list"]:
                        obj_ori_start_offset = obj["segments"]["start_offset"]
                        obj_ori_end_offset = obj["segments"]["end_offset"]
                        try:
                            obj_start_idx = offset_dict[obj_ori_start_offset]
                            obj_end_idx = offset_dict[obj_ori_end_offset] - 1
                        except:
                            common.logger.info(all_tokens_with_orig_offset)
                            common.logger.info(offset_dict)
                            assert example['text'][obj_ori_start_offset:
                                                   obj_ori_end_offset] == obj['text']
                            common.logger.info("{}".format(
                                example['text'][obj_ori_start_offset:obj_ori_end_offset]))
                            common.logger.info("{},{}/ {},{}".format(
                                obj_ori_start_offset, obj_ori_end_offset, obj_start_idx, obj_end_idx))
                            errmsg = "obj '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                                obj['text'], ' '.join(tokens), example['text'])
                            common.logger.warn(errmsg)
                            continue
                        if obj_end_idx >= seq_len:
                            errmsg = "obj {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                                obj_end_idx, obj['text'], ' '.join(tokens), example['text'])
                            common.logger.warn(errmsg)
                            continue
                        stat_info['relation_cnt'] += 1
                        relation_id = second_label2id[obj['label']]
                        second_label_start_ids[obj_start_idx][relation_id] = 1
                        second_label_end_ids[obj_end_idx][relation_id] = 1
            else:
                for triple in example["triple_list"]:
                    subject_ori_start_offset = triple["subject"]["segments"]["start_offset"]
                    subject_ori_end_offset = triple["subject"]["segments"]["end_offset"]
                    try:
                        subject_start_idx = offset_dict[subject_ori_start_offset]
                        subject_end_idx = offset_dict[subject_ori_end_offset] - 1
                    except:
                        errmsg = "subject '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            triple["subject"]['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if subject_end_idx >= seq_len:
                        errmsg = "subject {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            subject_end_idx, triple["subject"]['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    stat_info['relation_cnt'] += 1

                    first_label_start_ids[subject_start_idx] = 1
                    first_label_end_ids[subject_end_idx] = 1

                    obj = triple['object']
                    obj_ori_start_offset = obj["segments"]["start_offset"]
                    obj_ori_end_offset = obj["segments"]["end_offset"]
                    try:
                        obj_start_idx = offset_dict[obj_ori_start_offset]
                        obj_end_idx = offset_dict[obj_ori_end_offset] - 1
                    except:
                        errmsg = "obj '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            obj['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if obj_end_idx >= seq_len:
                        errmsg = "obj {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            obj_end_idx, obj['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    rel_id = second_label2id[triple['relation']]

                    key = (example['id'], int(subject_start_idx),
                           int(subject_end_idx), 0)
                    if key not in object_dict:
                        object_dict[key] = []
                    object_dict[key].append(common.LabelSpan(
                        start_idx=obj_start_idx, end_idx=obj_end_idx, label_id=rel_id))
            tokens_ids = np.array(tokens_ids)
            token_type_ids = np.array(token_type_ids)
            input_mask = np.array(input_mask)
            extra_mask = np.array(extra_mask)
            first_label_start_ids = first_label_start_ids.reshape(-1, 1)
            first_label_end_ids = first_label_end_ids.reshape(-1, 1)
            assert len(tokens_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(token_type_ids) == max_seq_len
            assert len(first_label_start_ids) == max_seq_len
            assert len(first_label_end_ids) == max_seq_len
            assert len(second_label_start_ids) == max_seq_len
            assert len(second_label_end_ids) == max_seq_len
            assert len(relativate_pos_label_ids) == max_seq_len
            features.append(InputFeatures(example_id=example['id'], tokens_ids=tokens_ids, input_mask=input_mask, seq_len=seq_len,
                                          token_type_ids=token_type_ids,
                                          second_label_mask=second_label_mask,
                                          first_label_start_ids=first_label_start_ids,
                                          first_label_end_ids=first_label_end_ids,
                                          first_label_start=first_label_start,
                                          first_label_end=first_label_end,
                                          second_label_start_ids=second_label_start_ids,
                                          second_label_end_ids=second_label_end_ids,
                                          golden_label=None if data_type == 'train' else example[
                                              'triple_list'],
                                          ori_tokens=ori_tokens,
                                          #   relative_pos_label=relativate_pos_label_ids,
                                          extra_mask=extra_mask))
            # token_ids = torch.tensor(token_ids, dtype=torch.long)
            # token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            # input_mask = torch.tensor(input_mask, dtype=torch.float32)
            # features.append(InputFeatures(example_id=example['id'], tokens_ids=token_ids, input_mask=input_mask, seq_len=seq_len,
            #                               token_type_ids=token_type_ids,
            #                               first_label_start_ids=first_label_start_ids,
            #                               first_label_end_ids=first_label_end_ids,
            #                               first_label_start=first_label_start,
            #                               first_label_end=first_label_end,
            #                               second_label_start_ids=second_label_start_ids,
            #                               second_label_end_ids=second_label_end_ids,
            #                               golden_label=None if data_type == 'train' else example[
            #                                   'triple_list']))

        if data_type != "train":
            return {
                'features': features, 'second_label_dict': object_dict, 'stat_info': stat_info
            }
        return {'features': features, "stat_info": stat_info}

    """Processor for the event extraction data set."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if 'id' not in line:
                line['id'] = guid
            examples.append(line)
        return examples

    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_len, first_label_map_file=None, second_label_map_file=None, data_type="train", pad_id=0, do_lower_case=False, add_event_type=False, use_event_str=False):
        # logger.info(first_label_map_file)
        first_label2id = self.get_labels(label_file=first_label_map_file)[1]
        # logger.info(first_label2id)
        second_label2id = self.get_labels(label_file=second_label_map_file)[1]
        features = []

        stat_info = {"trigger_cnt": 0, "argument_cnt": 0}
        if data_type != "train":
            argument_dict = {}

        for example in examples:
            text = example['text']
            tokens, ori_tokens = [], []
            all_tokens_with_orig_offset = []
            offset = 0
            for token in basic_tokenize(text):
                sub_offset = 0
                if token != ' ':  # skip space
                    ori_tokens.append(token)

                token_lower = token.lower() if do_lower_case else token
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                    proc_sub_token = sub_token
                    if sub_token.startswith('##'):
                        proc_sub_token = sub_token[2:]
                    new_sub_offset = token_lower.find(
                        proc_sub_token, sub_offset)
                    if new_sub_offset > -1:
                        sub_offset = new_sub_offset
                    all_tokens_with_orig_offset.append(
                        (sub_token, offset + sub_offset, token, offset))
                    if new_sub_offset > -1:
                        sub_offset += len(sub_token)
                    else:
                        sub_offset += 1
                offset += len(token)
            if add_event_type and use_event_str:
                event_tokens_ids = [1]
                for event_type in first_label2id.keys():
                    event_tokens_ids += tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(event_type)) + [1]
                event_tokens_ids = event_tokens_ids[1:-1]
                self._truncate_seq_pair(
                    tokens, event_tokens_ids, max_seq_len - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2".
                special_tokens_count = 2
                if len(tokens) > max_seq_len - special_tokens_count:
                    tokens = tokens[: max_seq_len - special_tokens_count]
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[: max_seq_len - special_tokens_count]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            seq_len = len(tokens)
            token_type_ids = [0] * seq_len
            ori_tokens = [tokenizer.cls_token] + \
                ori_tokens + [tokenizer.sep_token]

            all_tokens_with_orig_offset = [
                ('CLS', 0, '', 0)] + all_tokens_with_orig_offset

            offset_dict = {}
            for i, (_, _, orig_token, offset) in enumerate(all_tokens_with_orig_offset):
                if i == 0:
                    continue
                offset_dict.setdefault(offset, i)
                offset_dict.setdefault(offset + len(orig_token), i + 1)

            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

            if add_event_type and use_event_str:
                # tokens.extend(event_tokens)
                # tokens.append(tokenizer.sep_token)
                tokens_ids.extend(event_tokens_ids + [102])
                token_type_ids += [1] * (len(event_tokens_ids) + 1)

            extended_seq_len = len(tokens_ids)

            input_mask = [1] * extended_seq_len
            extra_mask = [0] * extended_seq_len

            padding_length = max_seq_len - len(tokens_ids)
            tokens_ids += [pad_id] * padding_length
            token_type_ids += [pad_id] * padding_length
            input_mask += [pad_id] * padding_length
            extra_mask += [1] * padding_length

            first_label_start_ids = np.zeros(
                (max_seq_len, len(first_label2id)))
            first_label_end_ids = np.zeros((max_seq_len, len(first_label2id)))
            second_label_start_ids = np.zeros(
                (max_seq_len, len(second_label2id)))
            second_label_end_ids = np.zeros(
                (max_seq_len, len(second_label2id)))
            first_label_start, first_label_end = np.zeros(
                max_seq_len), np.zeros(max_seq_len)
            text_type_ids = np.zeros(max_seq_len)
            first_label_ids = np.zeros(len(first_label2id))

            second_label_mask = 1.0

            if data_type == "train":
                for first_label in example["trigger_list"]:
                    first_ori_start_offset = first_label["segments"]["start_offset"]
                    first_ori_end_offset = first_label["segments"]["end_offset"]
                    try:
                        first_start_idx = offset_dict[first_ori_start_offset]
                        first_end_idx = offset_dict[first_ori_end_offset] - 1
                    except:
                        # logger.warn(tokens)
                        # logger.warn("{},{},{}".format(
                        #     text[first_ori_start_offset:first_ori_end_offset+1], first_ori_start_offset, first_ori_end_offset))
                        errmsg = "trigger '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            first_label['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if first_end_idx >= seq_len:
                        errmsg = "trigger '{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            first_label['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    # print(first_label2id)
                    trigger_id = first_label2id[first_label['label']]
                    first_label_ids[trigger_id] = 1
                    first_label_start_ids[first_start_idx][trigger_id] = 1
                    first_label_end_ids[first_end_idx][trigger_id] = 1

                selected_event = example["event"]
                if len(selected_event) == 0:
                    assert len(example["trigger_list"]) == 0
                    second_label_mask = 0.0
                else:
                    trigger_ori_start_offset = example["event"]["trigger"]["segments"]["start_offset"]
                    trigger_ori_end_offset = example["event"]["trigger"]["segments"]["end_offset"]

                    try:
                        trigger_start_idx = offset_dict[trigger_ori_start_offset]
                        trigger_end_idx = offset_dict[trigger_ori_end_offset] - 1
                    except:
                        errmsg = "trigger '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            example["event"]["trigger"]["text"], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if trigger_end_idx >= seq_len:
                        # logger.info(tokens[trigger_end_idx])
                        errmsg = "trigger {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            trigger_end_idx, example["event"]["trigger"]['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    text_type_ids = np.array(
                        [0] * trigger_start_idx + [1] *
                        (trigger_end_idx - trigger_start_idx + 1) +
                        [0] * (max_seq_len - trigger_end_idx - 1))
                    stat_info['trigger_cnt'] += 1
                    first_label_start[trigger_start_idx] = 1
                    first_label_end[trigger_end_idx] = 1

                    for argument in example["event"]["arguments"]:
                        argument_ori_start_offset = argument["segments"]["start_offset"]
                        argument_ori_end_offset = argument["segments"]["end_offset"]
                        try:
                            argument_start_idx = offset_dict[argument_ori_start_offset]
                            argument_end_idx = offset_dict[argument_ori_end_offset] - 1
                        except:
                            errmsg = "argument '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                                argument['text'], ' '.join(tokens), example['text'])
                            common.logger.warn(errmsg)
                            continue
                        if argument_end_idx >= seq_len:
                            errmsg = "argument {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                                argument_end_idx, argument['text'], ' '.join(tokens), example['text'])
                            common.logger.warn(errmsg)
                            continue
                        stat_info['argument_cnt'] += 1
                        argument_id = second_label2id[argument['label']]
                        second_label_start_ids[argument_start_idx][argument_id] = 1
                        second_label_end_ids[argument_end_idx][argument_id] = 1
            else:
                for event in example["events"]:
                    trigger_ori_start_offset = event["trigger"]["segments"]["start_offset"]
                    trigger_ori_end_offset = event["trigger"]["segments"]["end_offset"]
                    try:
                        trigger_start_idx = offset_dict[trigger_ori_start_offset]
                        trigger_end_idx = offset_dict[trigger_ori_end_offset] - 1
                    except:
                        errmsg = "trigger '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                            event["trigger"]['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    if trigger_end_idx >= seq_len:
                        errmsg = "trigger {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                            trigger_end_idx, event["trigger"]['text'], ' '.join(tokens), example['text'])
                        common.logger.warn(errmsg)
                        continue
                    # assert event["trigger"]['text'] == example['text'][trigger_ori_start_offset:trigger_ori_end_offset]
                    # logger.info(' '.join(tokens))
                    # logger.info(
                    #     "{} --- {}".format(
                    #         ' '.join(tokens[trigger_start_idx:trigger_end_idx+1]), event["trigger"]['text']))
                    stat_info['trigger_cnt'] += 1
                    event_id = first_label2id[event["trigger"]['label']]
                    first_label_ids[event_id] = 1
                    key = (example['id'], int(trigger_start_idx),
                           int(trigger_end_idx), int(event_id))

                    first_label_start_ids[trigger_start_idx][event_id] = 1
                    first_label_end_ids[trigger_end_idx][event_id] = 1

                    event_argument_list = []
                    for argument in event["arguments"]:
                        argument_ori_start_offset = argument["segments"]["start_offset"]
                        argument_ori_end_offset = argument["segments"]["end_offset"]
                        try:
                            argument_start_idx = offset_dict[argument_ori_start_offset]
                            argument_end_idx = offset_dict[argument_ori_end_offset] - 1
                        except:
                            errmsg = "argument '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                                argument['text'], ' '.join(tokens), example['text'])
                            common.logger.warn(errmsg)
                            continue
                        if argument_end_idx >= seq_len:
                            errmsg = "argument {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                                argument_end_idx, argument['text'], ' '.join(tokens), example['text'])
                            common.logger.warn(errmsg)
                            continue
                        stat_info['argument_cnt'] += 1
                        argument_id = second_label2id[argument['label']]
                        event_argument_list.append(common.LabelSpan(
                            start_idx=argument_start_idx, end_idx=argument_end_idx, label_id=argument_id))
                    argument_dict[key] = event_argument_list
            tokens_ids = np.array(tokens_ids)
            token_type_ids = np.array(token_type_ids)
            input_mask = np.array(input_mask)
            assert len(tokens_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(token_type_ids) == max_seq_len
            assert len(first_label_start_ids) == max_seq_len
            assert len(first_label_end_ids) == max_seq_len
            assert len(second_label_start_ids) == max_seq_len
            assert len(second_label_end_ids) == max_seq_len
            assert len(text_type_ids) == max_seq_len
            features.append(InputFeatures(example_id=example['id'], tokens_ids=tokens_ids, input_mask=input_mask, seq_len=seq_len,
                                          token_type_ids=token_type_ids,
                                          second_label_mask=second_label_mask,
                                          first_label_start_ids=first_label_start_ids,
                                          first_label_end_ids=first_label_end_ids,
                                          first_label_start=first_label_start,
                                          first_label_end=first_label_end,
                                          second_label_start_ids=second_label_start_ids,
                                          second_label_end_ids=second_label_end_ids,
                                          golden_label=None if data_type == 'train' else example['events'],
                                          ori_tokens=ori_tokens,
                                          text_type_ids=text_type_ids,
                                          extra_mask=extra_mask,
                                          first_label_ids=first_label_ids,
                                          extended_seq_len=extended_seq_len))

        if data_type != "train":
            return {
                'features': features, 'second_label_dict': argument_dict, 'stat_info': stat_info
            }
        return {'features': features, "stat_info": stat_info}


class TriggerDataProcessor(DataProcessor):
    """Processor for the event detection data set."""

    def __init__(self, args, tokenizer) -> None:
        super(TriggerDataProcessor, self).__init__()
        self.max_seq_length = args.max_seq_length
        self.model_name = args.model_name
        self.label_file = args.first_label_file
        self.tokenizer = tokenizer
        self.padding_to_max = args.padding_to_max
        self.span_decode_strategy = args.span_decode_strategy
        self.label_str_file = args.label_str_file
        self.pretrain_model_name = args.pretrain_model_name

        self.id2label, self.label2id = self.load_labels()
        self.class_num = len(self.id2label)
        self.is_chinese = args.is_chinese
        self.data_type = args.data_type

        self.use_random_label_emb = args.use_random_label_emb
        self.use_label_embedding = args.use_label_embedding
        if self.use_label_embedding:
            self.label_ann_word_id_list_file = args.label_ann_word_id_list_file
            args.label_ann_vocab_size = len(self.load_json(args.label_ann_vocab_file))
        self.use_label_encoding = args.use_label_encoding
        self.label_list = args.label_list
        self.token_ids = None
        self.input_mask = None
        self.token_type_ids = None

    def convert_examples_to_feature(self, input_file, data_type, encode_label=True):
        assert not (data_type == 'train' and encode_label == False)

        if data_type == 'test' and self.data_type == 'maven':
            encode_label = False

        features, example_len_list = [], []
        stat_info = {"trigger_cnt": 0}
        examples = self.__load_examples(input_file, data_type)

        sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        # print(self.tokenizer.sep_token)

        for example in examples:
            encoded_results = self.tokenizer.encode_plus(example['text'], add_special_tokens=True, return_offsets_mapping=True)
            token_ids = encoded_results['input_ids']
            if self.pretrain_model_name == 'bert':
                token_type_ids = encoded_results['token_type_ids']
            input_mask = encoded_results['attention_mask']
            offsets = encoded_results['offset_mapping']
            offset_dict = self.__build_offset_mapping(offsets, len(token_ids))

            example_len_list.append(len(token_ids)-2)

            token_ids = token_ids[:self.max_seq_length]
            if self.pretrain_model_name == 'bert':
                token_type_ids = token_type_ids[:self.max_seq_length]
            input_mask = input_mask[:self.max_seq_length]

            if token_ids[-1] != sep_id:
                assert len(token_ids) == self.max_seq_length
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            token_ids = torch.tensor(token_ids, dtype=torch.long)
            if self.pretrain_model_name == 'bert':
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)
            if encode_label:
                results = self.encode_labels(example['events'], seq_len, offset_dict, token_ids)

                stat_info['trigger_cnt'] += results['trigger_cnt']
                assert len(token_ids) == len(input_mask) == len(results['trigger_starts']) == len(results['trigger_ends'])

                features.append(TriggerFeatures(example_id=example['id'],
                                                tokens_ids=token_ids,
                                                input_mask=input_mask,
                                                seq_len=seq_len,
                                                token_type_ids=token_type_ids if self.pretrain_model_name == 'bert' else None,
                                                first_label_start_ids=results['trigger_starts'],
                                                first_label_end_ids=results['trigger_ends'],
                                                golden_label=results['golden_labels']))
            else:
                seq_feature = TriggerFeatures(example_id=example['id'], tokens_ids=token_ids,
                                              token_type_ids=token_type_ids if self.pretrain_model_name == 'bert' else None, input_mask=input_mask, seq_len=seq_len)
                if self.data_type == 'maven':
                    results = self.__get_candidates4maven(example['events'], seq_len, offset_dict, token_ids)
                    seq_feature.golden_label = results['golden_labels']
                    stat_info['trigger_cnt'] += results['trigger_cnt']
                features.append(seq_feature)
        stat_info['max_token_len'] = max(example_len_list)
        stat_info['standard_deviation'] = np.std(example_len_list, ddof=1)
        return {'features': features, "stat_info": stat_info}

    def build_output_results(self, tokens, infers, goldens=None, ids=None):
        if self.data_type == 'defalut':
            return self.__build_output_results_default(tokens, infers, goldens)
        elif self.data_type == 'maven':
            return self.__build_output_results_maven(tokens, infers, goldens, ids)
        else:
            raise ValueError("no '{}' output_format.".format(self.data_type))

    def decode_label(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        if self.model_name == "LEAR" or self.model_name == "bert_span_ed":
            return self.__decode_label4span(batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids, is_logits)
        elif "crf" in self.model_name:
            return self.__decode_label4crf()

    def __build_output_results_default(self, tokens, infers, goldens):
        outputs = []
        for batch_idx, (token, seq_infers) in enumerate(zip(tokens, infers)):
            text = self.tokenizer.decode(token, skip_special_tokens=True)
            infer_list = [{'text': self.tokenizer.decode(token[infer.start_idx:infer.end_idx+1]),
                           'event_type':self.id2label[infer.label_id]} for infer in seq_infers]
            outputs.append({
                'text': text,
                'event_list_infer': infer_list
            })
            if goldens is not None:
                join_set = set(goldens[batch_idx]) & set(seq_infers)
                lack = set(goldens[batch_idx]) - join_set
                new = set(seq_infers) - join_set
                outputs[-1]['event_list_golden'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                                     'event_type':self.id2label[item.label_id]} for item in goldens[batch_idx]]
                outputs[-1]['lack'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                        'event_type':self.id2label[item.label_id]} for item in lack]
                outputs[-1]['new'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                       'event_type':self.id2label[item.label_id]} for item in new]
        return outputs

    def __build_output_results_maven(self, tokens, infers, goldens, ids):
        outputs = {}
        assert len(tokens) == len(infers) == len(goldens) == len(ids), "{} != {}".format(len(goldens), len(ids))
        # for batch_idx, (token, seq_infers, seq_goldens, doc_id) in enumerate(zip(tokens, infers, goldens, ids)):
        #     infer_label_dict = {(label.start_idx, label.end_idx): label.label_id for label in seq_infers}
        #     golden_label_dict = {(label.start_idx, label.end_idx): label.label_id for label in seq_goldens}
        #     cur_infer_labels = set([(label.start_idx, label.end_idx) for label in seq_infers])
        #     cur_golden_labels = set([(label.start_idx, label.end_idx) for label in seq_goldens])
        #     inner_labels = cur_infer_labels & cur_golden_labels
        #     sub_labels = cur_golden_labels - inner_labels
        #     seq_results = []
        #     for label in inner_labels:
        #         seq_results.append({'id': golden_label_dict[label], 'type_id': infer_label_dict[label] + 1})
        #     for label in sub_labels:
        #         seq_results.append({'id': golden_label_dict[label], 'type_id': 0})
        #     if doc_id not in outputs:
        #         outputs[doc_id] = []
        #     outputs[doc_id].extend(seq_results)
        for batch_idx, (token, seq_infers, seq_goldens, doc_id) in enumerate(zip(tokens, infers, goldens, ids)):
            # infer_label_dict = {(label.start_idx, label.end_idx): label.label_id for label in seq_infers}
            # golden_label_dict = {(label.start_idx, label.end_idx): label.label_id for label in seq_goldens}
            # cur_infer_labels = set([(label.start_idx, label.end_idx) for label in seq_infers])
            # cur_golden_labels = set([(label.start_idx, label.end_idx) for label in seq_goldens])
            # inner_labels = cur_infer_labels & cur_golden_labels
            # sub_labels = cur_golden_labels - inner_labels
            seq_results = []
            infer_label_dict = {(label.start_idx, label.end_idx): label.label_id for label in seq_infers}
            for candidate in seq_goldens:
                label_span = (candidate.start_idx, candidate.end_idx)
                if label_span in infer_label_dict:
                    seq_results.append({'id': candidate.label_id, 'type_id': infer_label_dict[label_span] + 1})
                else:
                    seq_results.append({'id': candidate.label_id, 'type_id': 0})
            # print(doc_id)
            doc_id = doc_id['doc_id']
            if doc_id not in outputs:
                outputs[doc_id] = []
            outputs[doc_id].extend(seq_results)
        outputs = [{'id': key, 'predictions': value}for key, value in outputs.items()]
        return outputs

    def __decode_label4span(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        if self.span_decode_strategy == "v5":
            return self.__extract_span_v5(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        elif self.span_decode_strategy == "v1":
            return self.__extract_span_v1(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        elif self.span_decode_strategy == "v5_fast":
            return self.__extract_span_v5_fast(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        else:
            raise ValueError("no {} span decoding strategy.".format(self.span_decode_strategy))

    def __decode_label4crf(self):
        pass

    def __extract_span_v5(self, starts, ends, seqlens=None, position_dict=None, scores=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        """extract span with the higher probability"""
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)
        if return_span_score:
            assert scores is not None
            span_score_list = [[] for _ in range(starts.shape[0])]
        if seqlens is not None:
            assert starts.shape[0] == len(seqlens)
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

                start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1
                for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                    if token_start_prob >= s_limit:
                        if end_idx != -1:  # build span
                            if return_span_score:
                                cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                           end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                            else:
                                cur_spans.append(common.LabelSpan(start_idx=start_idx,
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
                        cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                   end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                    else:
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=end_idx, label_id=label_idx))
                cur_spans = list(set(cur_spans))
                if return_cnt:
                    span_cnt += len(cur_spans)
                if return_span_score:
                    span_score_list[batch_idx].extend(
                        [(item.start_score, item.end_score) for item in cur_spans])
                    span_list[batch_idx].extend([common.LabelSpan(
                        start_idx=item.start_idx, end_idx=item.end_idx, label_id=item.label_id) for item in cur_spans])
                else:
                    span_list[batch_idx].extend(cur_spans)
        output = (span_list,)
        if return_cnt:
            output += (span_cnt,)
        if return_span_score:
            output += (span_score_list,)
        return output

    @staticmethod
    def extract_span_multi_process(label_num, batch_idx, seq_starts, seq_ends, seqlen, s_limit, e_limit, result_dict):
        cur_spans = []
        for label_idx in range(label_num):
            seq_start_labels = seq_starts[:, label_idx][:seqlen] if seqlen is not None else seq_starts[:, label_idx]
            seq_end_labels = seq_ends[:, label_idx][:seqlen] if seqlen is not None else seq_ends[:, label_idx]
            start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1
            for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                if token_start_prob >= s_limit:
                    if end_idx != -1:  # build span
                        cur_spans.append(common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_idx))
                        start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1  # reset state
                    if token_start_prob > start_prob:  # new start, if pre prob is lower, drop it
                        start_prob = token_start_prob
                        start_idx = token_idx
                if token_end_prob > e_limit and start_prob > s_limit:  # end
                    if token_end_prob > end_prob:
                        end_prob = token_end_prob
                        end_idx = token_idx
            if end_idx != -1:
                cur_spans.append(common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_idx))
        # print(len(cur_spans))
        # print("done: {}".format(batch_idx))
        result_dict[batch_idx] = list(set(cur_spans))

    def __extract_span_v5_fast(self, starts, ends, seqlens=None, position_dict=None, scores=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        """extract span with the higher probability"""
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)
        if seqlens is not None:
            assert starts.shape[0] == len(seqlens)
        if return_cnt:
            span_cnt = 0
        batch_size, label_num = starts.shape[0], starts.shape[-1]
        span_list = [[] for _ in range(starts.shape[0])]

        manager = Manager()
        # result_list = manager.list([[] for _ in range(starts.shape[0])])
        result_dict = manager.dict()
        p = Pool(cpu_count()//2)

        for batch_idx in range(batch_size):
            p.apply_async(self.extract_span_multi_process, args=(label_num, batch_idx,
                          starts[batch_idx, :, :], ends[batch_idx, :, :], None if seqlens is None else seqlens[batch_idx], s_limit, e_limit, result_dict))
        p.close()
        p.join()

        for batch_idx, spans in result_dict.items():
            if return_cnt:
                span_cnt += len(spans)
            span_list[batch_idx].extend(spans)
        output = (span_list,)
        if return_cnt:
            output += (span_cnt,)
        return output

    def __extract_span_v1(self, starts, ends, seqlens=None, position_dict=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        """extract span, which consists of the nearest <start position, end position> pair"""
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        if seqlens is not None:
            assert starts.shape[0] == seqlens.shape[0]
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)

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
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=cur_ends[0], label_id=int(label_idx)))
                cur_spans = list(set(cur_spans))
                span_list[batch_idx].extend(cur_spans)
        return (span_list,)

    def get_train_examples(self, input_file):
        """See base class."""
        return self.__create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self.__create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self.__create_examples(self._read_json(input_file), "test")

    def load_json(self, input_file):
        with open(input_file, 'r') as fr:
            loaded_data = json.load(fr)
        return loaded_data

    def load_labels(self):
        """See base class."""
        with open(self.label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def get_tokenizer(self):
        return self.tokenizer

    def get_class_num(self):
        return self.class_num

    def get_label_data(self, device, rebuild=False):
        if rebuild:
            self.token_ids = None
            self.input_mask = None
            self.token_type_ids = None
        if self.token_ids is None:
            if self.use_label_embedding:
                token_ids, input_mask = [], []
                max_len = 0
                with open(self.label_ann_word_id_list_file, 'r') as fr:
                    for line in fr.readlines():
                        if line != '\n':
                            token_ids.append([int(item) for item in line.strip().split(' ')])
                            max_len = max(max_len, len(token_ids[-1]))
                            input_mask.append([1] * len(token_ids[-1]))
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
            else:
                if self.use_label_encoding:
                    with open(self.label_list, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                else:
                    with open(self.label_str_file, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                token_ids, input_mask, max_len = [], [], 0
                for label_str in label_str_list:
                    encoded_results = self.tokenizer.encode_plus(label_str, add_special_tokens=True)
                    token_id = encoded_results['input_ids']
                    input_mask.append(encoded_results['attention_mask'])
                    max_len = max(max_len, len(token_id))
                    token_ids.append(token_id)
                assert max_len <= self.max_seq_length and len(token_ids) == self.class_num
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
                self.token_type_ids = torch.zeros_like(self.token_ids, dtype=torch.long).to(device)
        return self.token_ids, self.token_type_ids, self.input_mask

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def __create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if 'id' not in line:
                line['id'] = guid
            examples.append(line)
        return examples

    def __load_examples(self, input_file, data_type):
        examples = self.__create_examples(self._read_json(input_file), data_type)
        return examples

    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def encode_labels(self, events, seq_len, offset_dict, token_ids):
        first_label_start_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        first_label_end_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        golden_labels, trigger_cnt = [], 0
        for event in events:
            trigger_ori_start_offset = event["trigger"]["segments"]["start_offset"]
            trigger_ori_end_offset = event["trigger"]["segments"]["end_offset"]
            try:
                trigger_start_idx = offset_dict[trigger_ori_start_offset]
                trigger_end_idx = offset_dict[trigger_ori_end_offset]
            except:
                errmsg = "trigger '{}' doesn't exist in '{}'".format(
                    event["trigger"]['text'], self.tokenizer.decode(token_ids, skip_special_tokens=True))
                common.logger.warn(errmsg)
                continue
            if trigger_end_idx >= seq_len:
                errmsg = "trigger {}/'{}' beyond boundary, tokens is '{}'".format(
                    trigger_end_idx, event["trigger"]['text'], self.tokenizer.decode(token_ids, skip_special_tokens=True))
                common.logger.warn(errmsg)
                continue
            trigger_cnt += 1
            event_id = self.label2id[event["trigger"]['label']]
            first_label_start_ids[trigger_start_idx][event_id] = 1
            first_label_end_ids[trigger_end_idx][event_id] = 1
            golden_labels.append(common.LabelSpan(start_idx=trigger_start_idx, end_idx=trigger_end_idx, label_id=event_id))
        results = {
            'trigger_starts': first_label_start_ids,
            'trigger_ends': first_label_end_ids,
            'trigger_cnt': trigger_cnt,
            'golden_labels': golden_labels
        }
        return results

    def __get_candidates4maven(self, events, seq_len, offset_dict, token_ids):
        golden_labels, trigger_cnt = [], 0
        for event in events:
            trigger_ori_start_offset = event["trigger"]["segments"]["start_offset"]
            trigger_ori_end_offset = event["trigger"]["segments"]["end_offset"]
            try:
                trigger_start_idx = offset_dict[trigger_ori_start_offset]
                trigger_end_idx = offset_dict[trigger_ori_end_offset]
            except:
                errmsg = "trigger '{}' doesn't exist in '{}'".format(
                    event["trigger"]['text'], self.tokenizer.decode(token_ids, skip_special_tokens=True))
                common.logger.warn(errmsg)
                continue
            if trigger_end_idx >= seq_len:
                errmsg = "trigger {}/'{}' beyond boundary, tokens is '{}'".format(
                    trigger_end_idx, event["trigger"]['text'], self.tokenizer.decode(token_ids, skip_special_tokens=True))
                common.logger.warn(errmsg)
                continue
            candidate_id = event["trigger"]['label']
            trigger_cnt += 1
            golden_labels.append(common.LabelSpan(start_idx=trigger_start_idx, end_idx=trigger_end_idx, label_id=candidate_id))
        results = {
            'trigger_cnt': trigger_cnt,
            'golden_labels': golden_labels
        }
        return results

    def __build_offset_mapping(self, offsets, seq_Len):
        offset_dict = {}
        for token_idx in range(seq_Len):
            # skip [cls] and [sep]
            if token_idx == 0 or token_idx == (seq_Len - 1):
                continue
            token_start, token_end = offsets[token_idx]
            offset_dict[token_start] = token_idx
            offset_dict[token_end] = token_idx
        return offset_dict

    def __generate_batch(self, batch):
        batch_size = len(batch)
        batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
        ids = [f.example_id for f in batch]
        max_len = int(max(batch_seq_len))
        batch_golden_label = [f.golden_label for f in batch]

        batch_tokens_ids, batch_input_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)
        if self.pretrain_model_name == "bert":
            batch_token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        if batch[0].first_label_start_ids is not None:
            class_num = batch[0].first_label_start_ids.shape[-1]

            batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
                (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)

        for batch_idx in range(batch_size):
            batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]] = batch[batch_idx].tokens_ids
            if self.pretrain_model_name == "bert":
                batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]] = batch[batch_idx].token_type_ids
            batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]] = batch[batch_idx].input_mask
            if batch[0].first_label_start_ids is not None:
                batch_first_label_start_ids[batch_idx][:batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
                batch_first_label_end_ids[batch_idx][:batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids
        results = {'token_ids': batch_tokens_ids,
                   #    'token_type_ids': batch_token_type_ids,
                   'input_mask': batch_input_mask,
                   'seq_len': batch_seq_len,
                   'ids': ids,
                   'golden_label': batch_golden_label
                   }
        if self.pretrain_model_name == "bert":
            results['token_type_ids'] = batch_token_type_ids
        if batch[0].first_label_start_ids is not None:
            results['first_starts'] = batch_first_label_start_ids
            results['first_ends'] = batch_first_label_end_ids
            # results['golden_label'] = batch_golden_label
        return results

    def generate_batch_data(self):
        return self.__generate_batch


class TriggerCrfDataProcessor(DataProcessor):
    """Processor for the event extraction data set."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        label2id, id2label = {'X': 0}, {}
        # label2id, id2label = {}, {}
        for label_name in label2id_.keys():
            label2id['B-'+label_name] = len(label2id)
            label2id['I-'+label_name] = len(label2id)
        label2id['O'] = len(label2id)
        label2id['[START]'] = len(label2id)
        label2id['[END]'] = len(label2id)
        for key, value in label2id.items():
            id2label[int(value)] = key
        return id2label, label2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if 'id' not in line:
                line['id'] = guid
            examples.append(line)
        return examples

    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def _add_label_str(self, trigger2id):
        return ' [unused1] '.join(trigger2id.keys())

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_len, first_label_map_file=None, second_label_map_file=None, data_type="train", pad_id=0, do_lower_case=False, add_event_type=False, is_maven=False):
        first_label2id = self.get_labels(label_file=first_label_map_file)[1]
        features = []
        stat_info = {"trigger_cnt": 0}
        for example in examples:
            encoded_results = tokenizer.encode(example['text'], add_special_tokens=True)
            tokens = encoded_results.tokens
            token_ids = encoded_results.ids
            token_type_ids = encoded_results.type_ids
            input_mask = encoded_results.attention_mask

            offset_dict = {}
            for token_idx in range(len(tokens)-1):  # skip [sep]
                if token_idx == 0:  # skip [cls]
                    continue
                token_start, token_end = encoded_results.offsets[token_idx]
                offset_dict[token_start] = token_idx
                offset_dict[token_end] = token_idx

            token_ids = token_ids[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]
            input_mask = input_mask[:max_seq_len]

            sep_id = tokenizer.token_to_id('[SEP]')
            if token_ids[-1] != sep_id:
                assert len(token_ids) == max_seq_len
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            input_mask = [1] * len(token_ids)

            label_ids = torch.zeros(seq_len, dtype=torch.float32)
            label_ids.fill_(first_label2id['O'])
            golden_labels = []

            for event in example["events"]:
                trigger_ori_start_offset = event["trigger"]["segments"]["start_offset"]
                trigger_ori_end_offset = event["trigger"]["segments"]["end_offset"]
                try:
                    trigger_start_idx = offset_dict[trigger_ori_start_offset]
                    trigger_end_idx = offset_dict[trigger_ori_end_offset]
                except:
                    errmsg = "trigger '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                        event["trigger"]['text'], ' '.join(tokens), example['text'])
                    common.logger.warn(errmsg)
                    continue
                if trigger_end_idx >= seq_len:
                    errmsg = "trigger {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                        trigger_end_idx, event["trigger"]['text'], ' '.join(tokens), example['text'])
                    common.logger.warn(errmsg)
                    continue
                if not (is_maven and data_type == 'test'):
                    assert ''.join(tokens[trigger_start_idx:trigger_end_idx+1]).replace("##",
                                                                                        "").lower() == event['trigger']['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[trigger_start_idx:trigger_end_idx+1]).replace("##", "").lower(), event['trigger']['text'].lower().replace(" ", ""))
                stat_info['trigger_cnt'] += 1
                # print(is_maven)
                # print(data_type)
                # if is_maven and data_type == 'test':
                #     event_id = event['trigger']['label']
                # else:

                label_ids[trigger_start_idx] = first_label2id['B-' +
                                                              event["trigger"]['label']]
                if trigger_start_idx != trigger_end_idx:
                    for idx in range(trigger_start_idx+1, trigger_end_idx+1):
                        label_ids[idx] = first_label2id['I-' +
                                                        event["trigger"]['label']]

                golden_labels.append(common.LabelSpan(
                    start_idx=trigger_start_idx, end_idx=trigger_end_idx, label_id=event["trigger"]['label']))

            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)
            assert len(token_ids) == len(input_mask) == len(token_type_ids) == len(
                label_ids)

            features.append(TriggerFeatures(example_id=example['id'],
                                            tokens_ids=token_ids,
                                            input_mask=input_mask,
                                            seq_len=seq_len,
                                            token_type_ids=token_type_ids,
                                            first_label_start_ids=label_ids,
                                            first_label_end_ids=None,
                                            golden_label=golden_labels))
        return {'features': features, "stat_info": stat_info}


class TriggerSoftmaxDataProcessor(DataProcessor):
    """Processor for the event extraction data set."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        # label2id, id2label = {'X': 0}, {}
        label2id, id2label = {}, {}
        for label_name in label2id_.keys():
            label2id['B-'+label_name] = len(label2id)
            label2id['I-'+label_name] = len(label2id)
        label2id['O'] = len(label2id)
        # label2id['[START]'] = len(label2id)
        # label2id['[END]'] = len(label2id)
        for key, value in label2id.items():
            id2label[int(value)] = key
        return id2label, label2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if 'id' not in line:
                line['id'] = guid
            examples.append(line)
        return examples

    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def _add_label_str(self, trigger2id):
        return ' [unused1] '.join(trigger2id.keys())

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_len, first_label_map_file=None, second_label_map_file=None, data_type="train", pad_id=0, do_lower_case=False, add_event_type=False, is_maven=False):
        first_label2id = self.get_labels(label_file=first_label_map_file)[1]

        # word2id = np.load(args.event_word2id_file)
        # event_str_ids = []

        # event_type_idx, start = [], 0
        # for event_type in event_type_list:
        #     cleaned_text = clean_str(event_type)
        #     cur_words = list(map(lambda x: x.split(), cleaned_text))
        #     event_type_idx.append((start, start+len(cur_words)))
        #     start += len(cur_words)
        #     event_type_idx.extend([word2id[word] for word in cur_words])

        features = []

        stat_info = {"trigger_cnt": 0}

        event_type_str = []
        # vocab_set = set()

        for example in examples:

            encoded_results = tokenizer.encode(
                example['text'], add_special_tokens=True)
            tokens = encoded_results.tokens
            token_ids = encoded_results.ids
            token_type_ids = encoded_results.type_ids
            input_mask = encoded_results.attention_mask

            offset_dict = {}
            for token_idx in range(len(tokens)-1):  # skip [sep]
                if token_idx == 0:  # skip [cls]
                    continue
                token_start, token_end = encoded_results.offsets[token_idx]
                offset_dict[token_start] = token_idx
                offset_dict[token_end] = token_idx

            token_ids = token_ids[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]
            input_mask = input_mask[:max_seq_len]

            sep_id = tokenizer.token_to_id('[SEP]')
            if token_ids[-1] != sep_id:
                assert len(token_ids) == max_seq_len
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            input_mask = [1] * len(token_ids)

            label_ids = torch.zeros(seq_len, dtype=torch.float32)
            label_ids.fill_(first_label2id['O'])
            golden_labels = []

            for event in example["events"]:
                trigger_ori_start_offset = event["trigger"]["segments"]["start_offset"]
                trigger_ori_end_offset = event["trigger"]["segments"]["end_offset"]
                try:
                    trigger_start_idx = offset_dict[trigger_ori_start_offset]
                    trigger_end_idx = offset_dict[trigger_ori_end_offset]
                except:
                    errmsg = "trigger '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                        event["trigger"]['text'], ' '.join(tokens), example['text'])
                    common.logger.warn(errmsg)
                    continue
                if trigger_end_idx >= seq_len:
                    errmsg = "trigger {}/'{}' beyond boundary, tokens is '{}'\noriginal sentence is:{}".format(
                        trigger_end_idx, event["trigger"]['text'], ' '.join(tokens), example['text'])
                    common.logger.warn(errmsg)
                    continue
                if not (is_maven and data_type == 'test'):
                    assert ''.join(tokens[trigger_start_idx:trigger_end_idx+1]).replace("##",
                                                                                        "").lower() == event['trigger']['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[trigger_start_idx:trigger_end_idx+1]).replace("##", "").lower(), event['trigger']['text'].lower().replace(" ", ""))
                stat_info['trigger_cnt'] += 1
                # print(is_maven)
                # print(data_type)
                # if is_maven and data_type == 'test':
                #     event_id = event['trigger']['label']
                # else:
                for idx in range(trigger_start_idx, trigger_end_idx):
                    assert label_ids[idx] == first_label2id['O']
                label_ids[trigger_start_idx] = first_label2id['B-' +
                                                              event["trigger"]['label']]
                if trigger_start_idx != trigger_end_idx:
                    for idx in range(trigger_start_idx+1, trigger_end_idx+1):
                        label_ids[idx] = first_label2id['I-' +
                                                        event["trigger"]['label']]
                # event_id = first_label2id[event["trigger"]['label']]

                golden_labels.append(common.LabelSpan(
                    start_idx=trigger_start_idx, end_idx=trigger_end_idx, label_id=event["trigger"]['label']))

            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)
            assert len(token_ids) == len(input_mask) == len(token_type_ids) == len(
                label_ids)

            features.append(TriggerFeatures(example_id=example['id'],
                                            tokens_ids=token_ids,
                                            input_mask=input_mask,
                                            seq_len=seq_len,
                                            token_type_ids=token_type_ids,
                                            first_label_start_ids=label_ids,
                                            first_label_end_ids=None,
                                            golden_label=golden_labels))
        return {'features': features, "stat_info": stat_info}


class NerDataProcessor(DataProcessor):
    """Processor for the named entity recognition data set."""

    def __init__(self, args, tokenizer) -> None:
        super(NerDataProcessor, self).__init__()
        self.max_seq_length = args.max_seq_length
        self.model_name = args.model_name
        self.label_file = args.first_label_file
        self.tokenizer = tokenizer
        self.padding_to_max = args.padding_to_max
        self.is_nested = args.exist_nested
        self.span_decode_strategy = args.span_decode_strategy
        self.label_str_file = args.label_str_file

        self.id2label, self.label2id = self.load_labels()
        self.class_num = len(self.id2label)
        self.is_chinese = args.is_chinese

        self.use_random_label_emb = args.use_random_label_emb
        self.use_label_embedding = args.use_label_embedding
        if self.use_label_embedding:
            self.label_ann_word_id_list_file = args.label_ann_word_id_list_file
            args.label_ann_vocab_size = len(self.load_json(args.label_ann_vocab_file))
        self.use_label_encoding = args.use_label_encoding
        self.label_list = args.label_list
        self.token_ids = None
        self.input_mask = None
        self.token_type_ids = None

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def load_json(self, input_file):
        with open(input_file, 'r') as fr:
            loaded_data = json.load(fr)
        return loaded_data

    def load_labels(self):
        """See base class."""
        with open(self.label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def get_tokenizer(self):
        return self.tokenizer

    def get_class_num(self):
        return self.class_num

    def get_label_data(self, device, rebuild=False):
        if rebuild:
            self.token_ids = None
            self.input_mask = None
            self.token_type_ids = None
        if self.token_ids is None:
            if self.use_label_embedding:
                token_ids, input_mask = [], []
                max_len = 0
                with open(self.label_ann_word_id_list_file, 'r') as fr:
                    for line in fr.readlines():
                        if line != '\n':
                            token_ids.append([int(item) for item in line.strip().split(' ')])
                            max_len = max(max_len, len(token_ids[-1]))
                            input_mask.append([1] * len(token_ids[-1]))
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
            else:
                if self.use_label_encoding:
                    with open(self.label_list, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                else:
                    with open(self.label_str_file, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                token_ids, input_mask, max_len = [], [], 0
                for label_str in label_str_list:
                    encoded_results = self.tokenizer.encode_plus(label_str, add_special_tokens=True)
                    token_id = encoded_results['input_ids']
                    input_mask.append(encoded_results['attention_mask'])
                    max_len = max(max_len, len(token_id))
                    token_ids.append(token_id)
                assert max_len <= self.max_seq_length and len(token_ids) == self.class_num
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
                self.token_type_ids = torch.zeros_like(self.token_ids, dtype=torch.long).to(device)
        return self.token_ids, self.token_type_ids, self.input_mask

    def decode_label4span(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        if self.is_nested:
            assert batch_match_label_ids is not None
            return self._extract_nested_span(batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids, is_logits)
        elif self.span_decode_strategy == "v5":
            return self._extract_span_v5(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        elif self.span_decode_strategy == "v1":
            return self._extract_span_v1(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        else:
            raise ValueError("no {} span decoding strategy.".format(self.span_decode_strategy))

    def decode_label4crf(self):
        pass

    def decode_label(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        if self.model_name == "SERS" or self.model_name == "bert_ner":
            return self.decode_label4span(batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids, is_logits)
        elif "crf" in self.model_name:
            return self.decode_label4crf()

    def encode_labels(self, entities, seq_len, offset_dict, tokens):
        first_label_start_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        first_label_end_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        if self.is_nested:
            match_label = torch.zeros((seq_len, seq_len, self.class_num), dtype=torch.float32)
        golden_labels, entity_cnt = [], 0

        for label in entities:
            label_start_offset = label["start_offset"]
            label_end_offset = label["end_offset"]
            try:
                start_idx = offset_dict[label_start_offset]
                end_idx = offset_dict[label_end_offset]
            except:
                # logger.warn(tokens)
                # logger.warn("{},{},{}".format(
                #     text[label_start_offset:label_end_offset+1], label_start_offset, label_end_offset))
                errmsg = "first_label '{}' doesn't exist in '{}'\n".format(label['text'], ' '.join(tokens))
                common.logger.warn(errmsg)
                continue
            if end_idx >= seq_len:
                continue
            if not self.is_chinese:
                assert ''.join(tokens[start_idx:end_idx+1]).replace("##",
                                                                    "").lower() == label['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[start_idx:end_idx+1]).replace("##", "").lower(), label['text'].lower().replace(" ", ""))
            entity_cnt += 1
            label_id = self.label2id[label['label']]
            golden_labels.append(common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))

            first_label_start_ids[start_idx][label_id] = 1
            first_label_end_ids[end_idx][label_id] = 1
            if self.is_nested:
                match_label[start_idx, end_idx, label_id] = 1
        results = {
            'entity_starts': first_label_start_ids,
            'entity_ends': first_label_end_ids,
            'entity_cnt': entity_cnt,
            'golden_labels': golden_labels
        }
        if self.is_nested:
            results['match_label'] = match_label
        return results

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line['id'] = guid
            examples.append(line)
        return examples

    @ classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def load_examples(self, input_file, data_type):
        examples = self._create_examples(self._read_json(input_file), data_type)
        return examples

    def build_offset_mapping(self, offsets, tokens):
        offset_dict = {}
        for token_idx in range(len(tokens)):
            # skip [cls] and [sep]
            if token_idx == 0 or token_idx == (len(tokens) - 1):
                continue
            token_start, token_end = offsets[token_idx]
            offset_dict[token_start] = token_idx
            offset_dict[token_end] = token_idx
        return offset_dict

    def convert_examples_to_feature(self, input_file, data_type):
        features = []
        stat_info = {'entity_cnt': 0, 'max_token_len': 0}
        examples = self.load_examples(input_file, data_type)
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        for example_idx, example in enumerate(examples):
            encoded_results = self.tokenizer.encode_plus(example['text'], add_special_tokens=True, return_offsets_mapping=True)
            token_ids = encoded_results['input_ids']
            token_type_ids = encoded_results['token_type_ids']
            input_mask = encoded_results['attention_mask']
            offsets = encoded_results['offset_mapping']
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            offset_dict = self.build_offset_mapping(offsets, tokens)

            stat_info['max_token_len'] = max(len(token_ids)-2, stat_info['max_token_len'])

            token_ids = token_ids[:self.max_seq_length]
            token_type_ids = token_type_ids[:self.max_seq_length]
            input_mask = input_mask[:self.max_seq_length]

            if token_ids[-1] != sep_id:
                assert len(token_ids) == self.max_seq_length
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            results = self.encode_labels(example['entities'], seq_len, offset_dict, tokens)
            stat_info['entity_cnt'] += results['entity_cnt']

            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)

            assert len(token_ids) == len(input_mask) == len(token_type_ids) == len(results['entity_starts']) == len(results['entity_ends'])

            if example_idx < 5:
                common.logger.info("*** Example ***")
                common.logger.info("guid: %s", example['id'])
                common.logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                common.logger.info("input_ids: %s", " ".join([str(x) for x in token_ids]))
                common.logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                common.logger.info("segment_ids: %s", " ".join([str(x) for x in token_type_ids]))
                common.logger.info("start_ids: %s" % " ".join([str(x) for x in results['entity_starts']]))
                common.logger.info("end_ids: %s" % " ".join([str(x) for x in results['entity_ends']]))

            features.append(NerFeatures(example_id=example['id'],
                                        tokens_ids=token_ids,
                                        input_mask=input_mask,
                                        seq_len=seq_len,
                                        token_type_ids=token_type_ids,
                                        first_label_start_ids=results['entity_starts'],
                                        first_label_end_ids=results['entity_ends'],
                                        golden_label=results['golden_labels'],
                                        match_label=results['match_label'] if self.is_nested else None))
        return {'features': features, "stat_info": stat_info}

    def build_seqlens_from_mask(self, input_mask):
        seqlens = [seq_mask.sum() for seq_mask in input_mask]
        return seqlens

    def _extract_span_v5(self, starts, ends, seqlens=None, position_dict=None, scores=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        # print(seqlens)
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)
        if return_span_score:
            assert scores is not None
            span_score_list = [[] for _ in range(starts.shape[0])]
        if seqlens is not None:
            assert starts.shape[0] == len(seqlens)
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

                start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1
                for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                    if token_start_prob >= s_limit:
                        if end_idx != -1:  # build span
                            if return_span_score:
                                cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                           end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                            else:
                                cur_spans.append(common.LabelSpan(start_idx=start_idx,
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
                        cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                   end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                    else:
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=end_idx, label_id=label_idx))
                cur_spans = list(set(cur_spans))
                if return_cnt:
                    span_cnt += len(cur_spans)
                if return_span_score:
                    span_score_list[batch_idx].extend(
                        [(item.start_score, item.end_score) for item in cur_spans])
                    span_list[batch_idx].extend([common.LabelSpan(
                        start_idx=item.start_idx, end_idx=item.end_idx, label_id=item.label_id) for item in cur_spans])
                else:
                    span_list[batch_idx].extend(cur_spans)
        output = (span_list,)
        if return_cnt:
            output += (span_cnt,)
        if return_span_score:
            output += (span_score_list,)
        return output

    def _extract_span_v1(self, starts, ends, seqlens=None, position_dict=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        if seqlens is not None:
            assert starts.shape[0] == seqlens.shape[0]
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)

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
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=cur_ends[0], label_id=int(label_idx)))
                cur_spans = list(set(cur_spans))
                span_list[batch_idx].extend(cur_spans)
        return (span_list,)

    def _extract_nested_span(self, starts, ends, seq_lens, matches, is_logits=True, limit=0.5):
        """ for nested"""
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)

        # [batch_size, seq_len]
        batch_size, seq_len, class_num = starts.size()

        # [batch_size, seq_len, class_num]
        extend_input_mask = torch.zeros_like(starts, dtype=torch.long)
        for batch_idx, seq_len_ in enumerate(seq_lens):
            extend_input_mask[batch_idx][:seq_len_] = 1

        start_label_mask = extend_input_mask.unsqueeze(
            -2).expand(-1, -1, seq_len, -1).bool()
        end_label_mask = extend_input_mask.unsqueeze(
            -3).expand(-1, seq_len, -1, -1).bool()

        # [batch_size, seq_len, seq_len, class_num]
        match_infer = matches > 0 if is_logits else matches > limit
        # [batch_size, seq_len, class_num]
        start_infer = starts > 0 if is_logits else starts > limit
        # [batch_size, seq_len, class_num]
        end_infer = ends > 0 if is_logits else ends > limit

        start_infer = start_infer.bool()
        end_infer = end_infer.bool()

        # match_infer = torch.ones_like(match_infer)

        match_infer = (
            match_infer & start_infer.unsqueeze(2).expand(-1, -1, seq_len, -1)
            & end_infer.unsqueeze(1).expand(-1, seq_len, -1, -1))

        match_label_mask = torch.triu((start_label_mask & end_label_mask).permute(
            0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
            batch_size, class_num, seq_len, seq_len).permute(0, 2, 3, 1)

        # [batch_size, seq_len, seq_len, class_num]
        match_infer = match_infer & match_label_mask

        span_list = [[] for _ in range(batch_size)]
        items = torch.where(match_infer == True)
        if len(items[0]) != 0:
            for idx in range(len(items[0])):
                batch_idx = int(items[0][idx])
                start_idx = int(items[1][idx])
                end_idx = int(items[2][idx])
                label_id = int(items[3][idx])
                span_list[batch_idx].append(
                    common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))
        return (span_list,)

    def build_output_results(self, tokens, infers, goldens=None):
        outputs = []
        for batch_idx, (token, seq_infers) in enumerate(zip(tokens, infers)):
            text = self.tokenizer.decode(token, skip_special_tokens=True)
            infer_list = [{'text': self.tokenizer.decode(token[infer.start_idx:infer.end_idx+1]),
                           'label':self.id2label[infer.label_id]} for infer in seq_infers]
            outputs.append({
                'text': text,
                'entity_infers': infer_list
            })
            if goldens is not None:
                join_set = set(goldens[batch_idx]) & set(seq_infers)
                lack = set(goldens[batch_idx]) - join_set
                new = set(seq_infers) - join_set
                outputs[-1]['entity_goldens'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                                  'label':self.id2label[item.label_id]} for item in goldens[batch_idx]]
                outputs[-1]['lack'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                        'label':self.id2label[item.label_id]} for item in lack]
                outputs[-1]['new'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                       'label':self.id2label[item.label_id]} for item in new]
        return outputs

    def _generate_batch(self, batch):
        batch_size, class_num = len(
            batch), batch[0].first_label_start_ids.shape[-1]

        batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
        ids = [f.example_id for f in batch]
        batch_golden_label = [f.golden_label for f in batch]
        max_len = int(max(batch_seq_len))

        batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
            (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

        batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
            (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)
        if self.is_nested:
            batch_match_label = torch.zeros(
                (batch_size, max_len, max_len, class_num), dtype=torch.float32)
        for batch_idx in range(batch_size):
            batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                        ] = batch[batch_idx].tokens_ids
            batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                            ] = batch[batch_idx].token_type_ids
            batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                        ] = batch[batch_idx].input_mask
            batch_first_label_start_ids[batch_idx][:
                                                   batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
            batch_first_label_end_ids[batch_idx][:
                                                 batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids
            if self.is_nested:
                batch_match_label[batch_idx, :batch[batch_idx].match_label.shape[0],
                                  :batch[batch_idx].match_label.shape[1]] = batch[batch_idx].match_label

        results = {'token_ids': batch_tokens_ids,
                   'token_type_ids': batch_token_type_ids,
                   'input_mask': batch_input_mask,
                   'seq_len': batch_seq_len,
                   'first_starts': batch_first_label_start_ids,
                   'first_ends': batch_first_label_end_ids,
                   'ids': ids,
                   'golden_label': batch_golden_label,
                   }
        if self.is_nested:
            results['match_label'] = batch_match_label
        return results

    def generate_batch_data(self):
        return self._generate_batch


class NerCrfDataProcessor(DataProcessor):
    """Processor for the event extraction data set."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        label2id, id2label = {'X': 0}, {}
        # label2id, id2label = {}, {}
        for label_name in label2id_.keys():
            label2id['B-'+label_name] = len(label2id)
            label2id['I-'+label_name] = len(label2id)
        label2id['O'] = len(label2id)
        label2id['[START]'] = len(label2id)
        label2id['[END]'] = len(label2id)
        for key, value in label2id.items():
            id2label[int(value)] = key
        return id2label, label2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line['id'] = guid
            examples.append(line)
        return examples

    @ classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_len, first_label_map_file=None, second_label_map_file=None, data_type="train", pad_id=0, do_lower_case=False, is_chinese=False):
        assert first_label_map_file is not None
        first_label2id = self.get_labels(label_file=first_label_map_file)[1]
        features = []
        stat_info = {'entity_cnt': 0}
        for example in examples:
            text = example['text']
            # stat_info['entity_cnt'] += len(example['entities'])

            encoded_results = tokenizer.encode(
                example['text'], add_special_tokens=True)
            tokens = encoded_results.tokens
            token_ids = encoded_results.ids
            token_type_ids = encoded_results.type_ids
            input_mask = encoded_results.attention_mask

            offset_dict = {}
            for token_idx in range(len(tokens)-1):  # skip [sep]
                if token_idx == 0:  # skip [cls]
                    continue
                token_start, token_end = encoded_results.offsets[token_idx]
                offset_dict[token_start] = token_idx
                offset_dict[token_end] = token_idx

            token_ids = token_ids[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]
            input_mask = input_mask[:max_seq_len]

            sep_id = tokenizer.token_to_id('[SEP]')
            if token_ids[-1] != sep_id:
                assert len(token_ids) == max_seq_len
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            # padding_length = max_seq_len - seq_len
            # token_ids += [pad_id] * padding_length
            # token_type_ids += [pad_id] * padding_length
            # input_mask += [pad_id] * padding_length

            label_ids = torch.zeros(seq_len, dtype=torch.float32)
            label_ids.fill_(first_label2id['O'])
            golden_labels = []

            for label in example["entities"]:
                label_start_offset = label["start_offset"]
                label_end_offset = label["end_offset"]
                try:
                    start_idx = offset_dict[label_start_offset]
                    end_idx = offset_dict[label_end_offset]
                except:
                    # common.logger.warn(tokens)
                    common.logger.warn("{},{},{}".format(
                        text[label_start_offset:label_end_offset+1], label_start_offset, label_end_offset))
                    errmsg = "first_label '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                        label['text'], ' '.join(tokens), example['text'])
                    common.logger.warn(errmsg)
                    continue
                if end_idx >= seq_len:
                    continue
                if not is_chinese:
                    assert ''.join(tokens[start_idx:end_idx+1]).replace("##",
                                                                        "").lower() == label['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[start_idx:end_idx+1]).replace("##", "").lower(), label['text'].lower().replace(" ", ""))
                stat_info['entity_cnt'] += 1
                label_ids[start_idx] = first_label2id['B-' + label['label']]
                if start_idx != end_idx:
                    for idx in range(start_idx+1, end_idx+1):
                        label_ids[idx] = first_label2id['I-' + label['label']]

                golden_labels.append(common.LabelSpan(
                    start_idx=start_idx, end_idx=end_idx, label_id=label['label']))
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)
            # assert len(
            #     token_ids) == max_seq_len, "{} - {}".format(len(token_ids), max_seq_len)
            # assert len(input_mask) == max_seq_len
            # assert len(token_type_ids) == max_seq_len
            # assert len(first_label_start_ids) == max_seq_len
            # assert len(first_label_end_ids) == max_seq_len
            assert len(token_ids) == len(input_mask) == len(token_type_ids)
            features.append(NerFeatures(example_id=example['id'],
                                        tokens_ids=token_ids,
                                        input_mask=input_mask,
                                        seq_len=seq_len,
                                        token_type_ids=token_type_ids,
                                        first_label_start_ids=label_ids,
                                        first_label_end_ids=None,
                                        golden_label=golden_labels,
                                        match_label=None))
        return features, stat_info


class NerSoftmaxDataProcessor(DataProcessor):
    """Processor for the event extraction data set."""

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def get_labels(self, label_file):
        """See base class."""
        with open(label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        # label2id, id2label = {'X': 0}, {}
        label2id, id2label = {}, {}
        for label_name in label2id_.keys():
            label2id['B-'+label_name] = len(label2id)
            label2id['I-'+label_name] = len(label2id)
        label2id['O'] = len(label2id)
        # label2id['[START]'] = len(label2id)
        # label2id['[END]'] = len(label2id)
        for key, value in label2id.items():
            id2label[int(value)] = key
        return id2label, label2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line['id'] = guid
            examples.append(line)
        return examples

    @ classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def convert_examples_to_feature(self, examples, tokenizer, max_seq_len, first_label_map_file=None, second_label_map_file=None, data_type="train", pad_id=0, do_lower_case=False, is_chinese=False):
        assert first_label_map_file is not None
        first_label2id = self.get_labels(label_file=first_label_map_file)[1]
        features = []
        stat_info = {'entity_cnt': 0}
        for example in examples:
            text = example['text']
            # stat_info['entity_cnt'] += len(example['entities'])

            encoded_results = tokenizer.encode(
                example['text'], add_special_tokens=True)
            tokens = encoded_results.tokens
            token_ids = encoded_results.ids
            token_type_ids = encoded_results.type_ids
            input_mask = encoded_results.attention_mask

            offset_dict = {}
            for token_idx in range(len(tokens)-1):  # skip [sep]
                if token_idx == 0:  # skip [cls]
                    continue
                token_start, token_end = encoded_results.offsets[token_idx]
                offset_dict[token_start] = token_idx
                offset_dict[token_end] = token_idx

            token_ids = token_ids[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]
            input_mask = input_mask[:max_seq_len]

            sep_id = tokenizer.token_to_id('[SEP]')
            if token_ids[-1] != sep_id:
                assert len(token_ids) == max_seq_len
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            # padding_length = max_seq_len - seq_len
            # token_ids += [pad_id] * padding_length
            # token_type_ids += [pad_id] * padding_length
            # input_mask += [pad_id] * padding_length

            label_ids = torch.zeros(seq_len, dtype=torch.float32)
            label_ids.fill_(first_label2id['O'])
            golden_labels = []

            for label in example["entities"]:
                label_start_offset = label["start_offset"]
                label_end_offset = label["end_offset"]
                try:
                    start_idx = offset_dict[label_start_offset]
                    end_idx = offset_dict[label_end_offset]
                except:
                    common.logger.warn(tokens)
                    common.logger.warn("{},{},{}".format(
                        text[label_start_offset:label_end_offset+1], label_start_offset, label_end_offset))
                    errmsg = "first_label '{}' doesn't exist in '{}'\noriginal sentence is:{}".format(
                        label['text'], ' '.join(tokens), example['text'])
                    common.logger.warn(errmsg)
                    continue
                if end_idx >= seq_len:
                    continue
                if not is_chinese:
                    assert ''.join(tokens[start_idx:end_idx+1]).replace("##",
                                                                        "").lower() == label['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[start_idx:end_idx+1]).replace("##", "").lower(), label['text'].lower().replace(" ", ""))
                stat_info['entity_cnt'] += 1
                label_ids[start_idx] = first_label2id['B-' + label['label']]
                if start_idx != end_idx:
                    for idx in range(start_idx+1, end_idx+1):
                        label_ids[idx] = first_label2id['I-' + label['label']]

                golden_labels.append(common.LabelSpan(
                    start_idx=start_idx, end_idx=end_idx, label_id=label['label']))
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)
            # assert len(
            #     token_ids) == max_seq_len, "{} - {}".format(len(token_ids), max_seq_len)
            # assert len(input_mask) == max_seq_len
            # assert len(token_type_ids) == max_seq_len
            # assert len(first_label_start_ids) == max_seq_len
            # assert len(first_label_end_ids) == max_seq_len
            assert len(token_ids) == len(input_mask) == len(token_type_ids)
            features.append(NerFeatures(example_id=example['id'],
                                        tokens_ids=token_ids,
                                        input_mask=input_mask,
                                        seq_len=seq_len,
                                        token_type_ids=token_type_ids,
                                        first_label_start_ids=label_ids,
                                        first_label_end_ids=None,
                                        golden_label=golden_labels,
                                        match_label=None))
        return features, stat_info


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids, ori_tokens=None, first_label_start_ids=None, first_label_end_ids=None,
                 first_label_start=None,
                 first_label_end=None,
                 second_label_start_ids=None,
                 second_label_end_ids=None, golden_label=None,  text_type_ids=None, relative_pos_label=None,
                 extra_mask=None, second_label_mask=1.0, first_label_ids=None, extended_seq_len=None, scores_ids=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.first_label_start = first_label_start
        self.first_label_end = first_label_end
        self.second_label_start_ids = second_label_start_ids
        self.second_label_end_ids = second_label_end_ids
        self.golden_label = golden_label
        self.second_label_mask = second_label_mask
        self.ori_tokens = ori_tokens
        self.text_type_ids = text_type_ids
        self.relative_pos_label = relative_pos_label
        self.extra_mask = extra_mask
        self.first_label_ids = first_label_ids
        self.extended_seq_len = extended_seq_len
        self.scores_ids = scores_ids
        self.match_label = match_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NerFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids,  first_label_start_ids=None, first_label_end_ids=None,
                 golden_label=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.golden_label = golden_label
        self.match_label = match_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TriggerFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids, first_label_start_ids=None, first_label_end_ids=None,
                 golden_label=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.golden_label = golden_label
        self.match_label = match_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn_relation(batch):
    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    max_len = int(max(batch_seq_len).item())
    batch_tokens_ids = torch.tensor(
        [f.tokens_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_input_mask = torch.tensor(
        [f.input_mask for f in batch], dtype=torch.float32)[:, :max_len]

    batch_token_type_ids = torch.tensor(
        [f.token_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_start_ids = torch.tensor([f.first_label_start_ids for f in batch], dtype=torch.float32)[:,
                                                                                                              :max_len]
    batch_first_label_end_ids = torch.tensor(
        [f.first_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_start = torch.tensor(
        [f.first_label_start for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_end = torch.tensor(
        [f.first_label_end for f in batch], dtype=torch.float32)[:, :max_len]
    batch_second_label_start_ids = torch.tensor([f.second_label_start_ids for f in batch], dtype=torch.float32, )[:,
                                                                                                                  :max_len]
    batch_second_label_end_ids = torch.tensor(
        [f.second_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    gold_label = [f.golden_label for f in batch]
    batch_second_label_mask = torch.tensor(
        [f.second_label_mask for f in batch], dtype=torch.float32)
    # batch_relative_pos_ids = torch.tensor(
    #     [f.relative_pos_label for f in batch], dtype=torch.long)[:, :max_len]
    batch_ori_tokens = [f.ori_tokens for f in batch]
    batch_extra_mask = torch.tensor(
        [f.extra_mask for f in batch], dtype=torch.long)[:, :max_len]
    ids = [f.example_id for f in batch]
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'first_start': batch_first_label_start,
            'first_end': batch_first_label_end,
            'second_starts': batch_second_label_start_ids,
            'second_ends': batch_second_label_end_ids,
            'second_label_mask': batch_second_label_mask,
            # 'relative_pos_ids': batch_relative_pos_ids,
            'golden_label': gold_label,
            'ori_tokens': batch_ori_tokens,
            'extra_mask': batch_extra_mask,
            'ids': ids}


def collate_fn_event(batch):
    batch_size = len(batch)

    event_str_ids = np.load(
        '/home/yangpan/workspace/onepass_ie/data/ace05/splited/event_type_uncased_whole_ids.npy', allow_pickle=True)
    # print(event_str_ids.shape)
    batch_event_str_ids = torch.tensor(
        np.repeat(event_str_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)
    # print(batch_event_str_ids.shape)

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    batch_extended_seq_len = [f.extended_seq_len for f in batch]
    # max_len = int(max(batch_extended_seq_len))
    max_len = int(max(batch_seq_len).item())
    batch_tokens_ids = torch.tensor(
        [f.tokens_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_input_mask = torch.tensor(
        [f.input_mask for f in batch], dtype=torch.float32)[:, :max_len]

    batch_token_type_ids = torch.tensor(
        [f.token_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_start_ids = torch.tensor([f.first_label_start_ids for f in batch], dtype=torch.float32)[:,
                                                                                                              :max_len]
    batch_first_label_end_ids = torch.tensor(
        [f.first_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_start = torch.tensor(
        [f.first_label_start for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_end = torch.tensor(
        [f.first_label_end for f in batch], dtype=torch.float32)[:, :max_len]
    batch_second_label_start_ids = torch.tensor([f.second_label_start_ids for f in batch], dtype=torch.float32, )[:,
                                                                                                                  :max_len]
    batch_second_label_end_ids = torch.tensor(
        [f.second_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    gold_label = [f.golden_label for f in batch]
    batch_second_label_mask = torch.tensor(
        [f.second_label_mask for f in batch], dtype=torch.float32)
    batch_text_type_ids = torch.tensor(
        [f.text_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_ori_tokens = [f.ori_tokens for f in batch]
    batch_extra_mask = torch.tensor(
        [f.extra_mask for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_ids = torch.tensor(
        [f.first_label_ids for f in batch], dtype=torch.float32)
    ids = [f.example_id for f in batch]
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'first_start': batch_first_label_start,
            'first_end': batch_first_label_end,
            'second_starts': batch_second_label_start_ids,
            'second_ends': batch_second_label_end_ids,
            'second_label_mask': batch_second_label_mask,
            'text_type_ids': batch_text_type_ids,
            'golden_label': gold_label,
            'ids': ids,
            'first_types': batch_first_label_ids,
            'extra_mask': batch_extra_mask,
            'ori_tokens': batch_ori_tokens, 'event_str_ids': batch_event_str_ids}
    #
    # batch_tokens_ids, batch_input_mask, batch_seq_len, batch_token_type_ids, batch_sub_head_ids, batch_sub_end_ids, batch_sub_head, batch_sub_end, \
    # batch_obj_head_ids, batch_obj_end_ids = map(torch.stack, zip(*batch))
    #
    # max_len = int(max(batch_seq_len).item())
    # # print(batch_seq_len)
    # batch_tokens_ids = batch_tokens_ids[:, :max_len]
    # batch_token_type_ids = batch_token_type_ids[:, :max_len]
    # batch_input_mask = batch_input_mask[:, :max_len]
    # # batch_seq_len = batch_seq_len[:, :max_len]
    # batch_sub_head_ids = batch_sub_head_ids[:, :max_len]
    # batch_sub_end_ids = batch_sub_end_ids[:, :max_len]
    # batch_sub_head = batch_sub_head[:, :max_len]
    # batch_sub_end = batch_sub_end[:, :max_len]
    # batch_obj_head_ids = batch_obj_head_ids[:, :max_len]
    # batch_obj_end_ids = batch_obj_end_ids[:, :max_len]
    # return {'token_ids': batch_tokens_ids,
    #         'token_type_ids': batch_token_type_ids,
    #         'input_mask': batch_input_mask,
    #         'seq_len': batch_seq_len,
    #         'first_heads': batch_sub_head_ids,
    #         'first_tails': batch_sub_end_ids,
    #         'sub_head': batch_sub_head,
    #         'sub_tail': batch_sub_end,
    #         'second_heads': batch_obj_head_ids,
    #         'second_tails': batch_obj_end_ids}


def collate_fn_classification(batch):
    batch_size = len(batch)
    # label_ids = np.load(
    #     '/data/yangpan/workspace/dataset/text_classification/imdb/emotional_orientation_uncased_whole_ids.npy', allow_pickle=True)

    # batch_label_ids = torch.tensor(
    #     np.repeat(label_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)
    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    # batch_extended_seq_len = [f.extended_seq_len for f in batch]
    max_len = int(max(batch_seq_len).item())
    batch_tokens_ids = torch.tensor(
        [f.tokens_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_input_mask = torch.tensor(
        [f.input_mask for f in batch], dtype=torch.float32)[:, :max_len]
    batch_token_type_ids = torch.tensor(
        [f.token_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_ids = torch.tensor(
        [f.first_label_ids for f in batch], dtype=torch.long)
    # batch_ori_tokens = [f.ori_tokens for f in batch]
    ids = [f.example_id for f in batch]
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'ids': ids,
            'first_types': batch_first_label_ids,
            # 'first_label_ids': batch_label_ids,
            # 'ori_tokens': batch_ori_tokens
            }


def collate_fn_multi_classification(batch):
    batch_size = len(batch)

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    # batch_label_ids = torch.tensor(
    #     [f.label_ids for f in batch], dtype=torch.float32)

    batch_label_ids = torch.stack([f.label_ids for f in batch], dim=0)

    # print(batch_label_ids.shape)
    # print(batch_tokens_ids)
    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        # print(batch[batch_idx].tokens_ids)
        # print(batch_tokens_ids[batch_idx])
        # print(batch_label_ids[batch_idx])
        # print('\n')
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_types': batch_label_ids,
            'ids': ids
            }


def collate_fn_trigger(batch):
    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
        (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)

    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
        batch_first_label_end_ids[batch_idx][:
                                             batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids

    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'ids': ids,
            'golden_label': batch_golden_label,
            }


def collate_fn_trigger_crf(batch):
    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids = torch.zeros(
        (batch_size, max_len), dtype=torch.long)

    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids

    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'ids': ids,
            'golden_label': batch_golden_label,
            }


def collate_fn_ner_crf(batch):

    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids = torch.zeros(
        (batch_size, max_len), dtype=torch.long)
    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'ids': ids,
            'golden_label': batch_golden_label
            }


def collate_fn_ner(batch):

    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
        (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)
    batch_match_label = torch.zeros(
        (batch_size, max_len, max_len, class_num), dtype=torch.float32)
    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
        batch_first_label_end_ids[batch_idx][:
                                             batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids
        batch_match_label[batch_idx, :batch[batch_idx].match_label.shape[0],
                          :batch[batch_idx].match_label.shape[1]] = batch[batch_idx].match_label

    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'ids': ids,
            'golden_label': batch_golden_label,
            'match_label': batch_match_label
            }

# def collate_fn_ner(batch):
#     batch_size = len(batch)

#     # label_ids = np.load(
#     #     '/data/yangpan/workspace/dataset/ner/conll03/processed/entity_with_annoation_uncased_whole_ids.npy', allow_pickle=True)

#     # batch_label_ids = torch.tensor(
#     #     np.repeat(label_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)

#     batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
#     max_len = int(max(batch_seq_len))
#     batch_tokens_ids = torch.tensor(
#         [f.tokens_ids for f in batch], dtype=torch.long)
#     batch_input_mask = torch.tensor(
#         [f.input_mask for f in batch], dtype=torch.float32)

#     batch_token_type_ids = torch.tensor(
#         [f.token_type_ids for f in batch], dtype=torch.long)

#     batch_first_label_start_ids = torch.tensor(
#         [f.first_label_start_ids for f in batch], dtype=torch.float32)
#     batch_first_label_end_ids = torch.tensor(
#         [f.first_label_end_ids for f in batch], dtype=torch.float32)
#     # batch_first_label_ids = torch.tensor(
#     #     [f.first_label_ids for f in batch], dtype=torch.float32)
#     batch_match_label = torch.tensor(
#         [f.match_label for f in batch], dtype=torch.float32)

#     # batch_ori_tokens = [f.ori_tokens for f in batch]
#     ids = [f.example_id for f in batch]

#     # batch_scores_ids = torch.tensor(
#     #     [f.scores_ids for f in batch], dtype=torch.float32)[:, : max_len]

#     label_num = batch_first_label_end_ids.shape[-1]
#     label_ids = np.array([idx for idx in range(label_num)], dtype=np.int)
#     batch_label_ids = torch.tensor(
#         np.repeat(label_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)
#     batch_golden_label = [f.golden_label for f in batch]

#     return {'token_ids': batch_tokens_ids,
#             'token_type_ids': batch_token_type_ids,
#             'input_mask': batch_input_mask,
#             'seq_len': batch_seq_len,
#             'first_starts': batch_first_label_start_ids,
#             'first_ends': batch_first_label_end_ids,
#             'first_label_ids': batch_label_ids,
#             'ids': ids,
#             # 'first_types': batch_first_label_ids,
#             # 'ori_tokens': batch_ori_tokens,
#             # 'scores_ids': batch_scores_ids,
#             'golden_label': batch_golden_label,
#             'match_label': batch_match_label}


class DataPreFetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].to(device=self.device,
                                                             non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
