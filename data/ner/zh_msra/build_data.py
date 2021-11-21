import json
import os
import random
from collections import OrderedDict

label_set = set()

label_ann_dict = {}


def load_data(input_file, max_seq_len=512):
    print("load {}".format(input_file))
    examples = []
    entity_cnt, max_len, cnt = 0, 0, 0
    data_dict = {}
    with open(input_file, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
        label_list = []
        b_label_cnt = 0
        pre_text = data[0]['context']
        for line in data:
            text = line["context"]
            words = text.split()
            # text = ' '.join(words)
            start_positions = line["start_position"]
            end_positions = line["end_position"]
            if pre_text != text:
                label_list = list(set(label_list))
                entity_cnt += len(label_list)
                # examples.append(OrderedDict(
                #     [
                #         ('text', pre_text),
                #         ('entities', label_list)
                #     ]
                # ))
                examples.append(OrderedDict(
                    [
                        ('text', pre_text),
                        ('entities', [
                            OrderedDict(
                                [
                                    ('label', label[0]),
                                    ('text', label[1]),
                                    ('start_offset', label[2]),
                                    ('end_offset', label[3])
                                ])
                            for label in label_list
                        ]
                        )
                    ]
                ))
                label_list = []
                pre_text = text
                max_len = max(max_len, len(text))
                if len(text) > max_seq_len:
                    cnt += 1
            if len(start_positions) == 0:
                continue
            entity_label = line['entity_label']
            label_set.add(entity_label)
            label_ann_dict[entity_label] = line['query']
            # add space offsets

            start_positions = [x + sum([len(w) for w in words[:x]])
                               for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]])
                             for x in end_positions]
            # entity_cnt += len(start_positions)
            for start, end in zip(start_positions, end_positions):
                # label_list.append(OrderedDict(
                #     [
                #         ('label', entity_label),
                #         ('text', text[start:end]),
                #         ('start_offset', start),
                #         ('end_offset', end)
                #     ]
                # ))
                label_list.append((entity_label, text[start:end], start, end))
        label_list = list(set(label_list))
        entity_cnt += len(label_list)
        examples.append(OrderedDict(
            [
                        ('text', pre_text),
                        ('entities', [
                            OrderedDict(
                                [
                                    ('label', label[0]),
                                    ('text', label[1]),
                                    ('start_offset', label[2]),
                                    ('end_offset', label[3])
                                ])
                            for label in label_list
                        ]
                        )
                        ]
        ))
    print("total {} examples, {} entities, max_len: {}, {} examples > {}".format(
        len(examples), entity_cnt, max_len, cnt, max_seq_len))
    return examples


def build_qa_format_data(examples, label2ann, data_type):
    qa_examples = []
    entity_cnt = 0
    for example in examples:
        entity_dict = {trigger: [] for trigger in label2ann.keys()}
        for entity in example['entities']:
            entity_dict[entity['label']].append(entity)
        for label_text, entities in entity_dict.items():
            entity_cnt += len(entities)
            qa_examples.append(
                OrderedDict(
                    [
                        ('text', example['text']),
                        ('query', label2ann[label_text]),
                        ('entities', entities)
                    ]
                )
            )
    print("[{}], total {} qa examples, {} entities".format(
        data_type, len(qa_examples), entity_cnt))
    return qa_examples


def build_few_shot_data(examples, label2id, selected_num=5):
    examples_dict = {idx: [] for idx in range(len(label2id))}
    selected = {}
    for example in examples:
        label_list = [entity['label'] for entity in example['entities']]
        label_set = set(label_list)
        for label in label_set:
            examples_dict[label2id[label]].append(example)
    fewshot_selected = []
    for idx, label_name in enumerate(label2id):
        cur_examples = []
        for cur_example in examples_dict[idx]:
            if cur_example['text'] not in selected:
                cur_examples.append(cur_example)
        s = random.sample(cur_examples, min(selected_num, len(cur_examples)))
        for item in s:
            selected[item['text']] = 0
        fewshot_selected.extend(s)
        print("{}: {}/{}".format(label_name, len(s), len(cur_examples)))
    return fewshot_selected


if __name__ == "__main__":
    train_examples = load_data("mrc-ner.train")
    dev_examples = load_data('mrc-ner.dev')
    test_examples = load_data('mrc-ner.test')

    print('total {} label'.format(len(label_set)))
    if not os.path.exists('./processed'):
        os.makedirs('./processed')
    with open('processed/train.json', 'w', encoding='utf-8') as fw:
        for example in train_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('processed/dev.json', 'w', encoding='utf-8') as fw:
        for example in dev_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('processed/test.json', 'w', encoding='utf-8') as fw:
        for example in test_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    label2id = {label: int(idx) for idx,
                label in enumerate(sorted(label_set))}
    id2label = {int(idx): label for idx,
                label in enumerate(sorted(label_set))}

    with open('processed/label_map.json', 'w', encoding='utf-8') as fw:
        json.dump([id2label, label2id], fw, indent=4, ensure_ascii=False)
    with open('processed/label_annotation.txt', 'w', encoding='utf-8') as fw:
        for idx in range(len(id2label)):
            fw.write("{}\n".format(label_ann_dict[id2label[idx]]))

    if not os.path.exists('./qa_format'):
        os.makedirs('./qa_format')
    qa_train_examples = build_qa_format_data(
        train_examples, label_ann_dict, 'train')
    qa_dev_examples = build_qa_format_data(
        dev_examples, label_ann_dict, 'dev')
    qa_test_examples = build_qa_format_data(
        test_examples, label_ann_dict, 'test')

    with open('qa_format/train.json', 'w', encoding='utf-8') as fw:
        for example in qa_train_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('qa_format/dev.json', 'w', encoding='utf-8') as fw:
        for example in qa_dev_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('qa_format/test.json', 'w', encoding='utf-8') as fw:
        for example in qa_test_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    if not os.path.exists('./fewshot'):
        os.makedirs('./fewshot')
    # few_examples = build_few_shot_data(train_examples, label2id)
    # with open('fewshot/train_5_shot.json', 'w', encoding='utf-8') as fw:
    #     for example in few_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # few_examples = random.sample(train_examples, int(len(train_examples)*0.1))
    # with open('fewshot/train_10_shot.json', 'w', encoding='utf-8') as fw:
    #     for example in few_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    for selected_cnt in [1, 5]:
        random.seed(1)
        few_examples = build_few_shot_data(
            train_examples, label2id, selected_num=selected_cnt)
        with open('fewshot/train_{}_shot.json'.format(selected_cnt), 'w', encoding='utf-8') as fw:
            for example in few_examples:
                fw.write(json.dumps(example, ensure_ascii=False) + '\n')
