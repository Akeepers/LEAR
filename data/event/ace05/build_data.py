import os
import json
import random
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from itertools import combinations, permutations

trigger_set = set()
argument_set = set()


def load_data(input_file, do_split=False, return_single=False, return_multi=False):
    print("load {}".format(input_file))
    examples, example_dict = [], {}

    event_cnt, argument_cnt = 0, 0

    with open(input_file, 'r', encoding='utf-8') as fr:
        # lines = json.load(fr)
        for line in tqdm(json.load(fr)):
            tokens = line['tokens']
            event_type = line['event_type']
            text = ' '.join(tokens)
            if text not in example_dict:
                example_dict[text] = []
            if event_type == "None":
                continue
            trigger_set.add(event_type)
            trigger_text = ' '.join(line['trigger_tokens'])
            start_idx, end_idx = line['trigger_start'], line['trigger_end']
            start_offset = start_idx + \
                sum([len(token) for token in tokens[:start_idx]])
            end_offset = end_idx + sum([len(token)
                                        for token in tokens[:end_idx+1]])
            assert trigger_text == text[start_offset:end_offset]
            trigger = OrderedDict(
                [
                    ('text', trigger_text),
                    ('label', event_type),
                    ('segments', {
                        'start_offset': start_offset,
                        'end_offset': end_offset
                    })
                ]
            )
            argument_list = []
            for entity in line['entities']:
                if entity['role'] == "None":
                    continue
                argument_set.add(entity['role'])
                argument_text = ' '.join(entity['tokens'])
                start_idx, end_idx = entity['idx_start'], entity['idx_end']
                start_offset = start_idx + \
                    sum([len(token) for token in tokens[:start_idx]])
                end_offset = end_idx + sum([len(token)
                                            for token in tokens[:end_idx+1]])
                assert argument_text == text[start_offset:end_offset]
                argument_list.append(OrderedDict(
                    [
                        ('text', argument_text),
                        ('label', entity['role']),
                        ('segments', {
                            'start_offset': start_offset,
                            'end_offset': end_offset
                        })
                    ]
                ))
            argument_cnt += len(argument_list)
            example_dict[text].append(OrderedDict([
                ('trigger', trigger),
                ('arguments', argument_list)
            ]))
    for text, events in example_dict.items():
        event_cnt += len(events)
        examples.append(OrderedDict(
            [
                ('text', text),
                ('events', events)
            ]
        ))
    print("total {} examples, {} events, {} arguments".format(
        len(examples), event_cnt, argument_cnt))
    # if do_split:
    #     print("total {} examples, {} events, {} arguments\n total {} splited examples, {} events, {} arguments".format(
    #         len(examples), event_cnt, argument_cnt, len(splited_examples), splited_event_cnt, splited_argument_cnt))
    # else:
    #     print("total {} examples, {} events, {} arguments".format(
    #         len(examples), event_cnt, argument_cnt))
    # if return_single:
    #     print("total {} single examples".format(len(single_examples)))
    # if return_multi:
    #     print("total {} multi examples".format(len(multi_examples)))
    return_dict = {"examples": examples}
    # if do_split:
    #     return_dict['splited_examples'] = splited_examples
    # if return_single:
    #     return_dict["single_examples"] = single_examples
    # if return_multi:
    #     return_dict["multi_examples"] = multi_examples
    return return_dict


def build_qa_format_data(examples, label2ann, data_type):
    qa_examples = []
    event_cnt = 0
    for example in examples:
        trigger_dict = {trigger: [] for trigger in label2ann.keys()}
        for event in example['events']:
            trigger = event['trigger']
            trigger_dict[trigger['label']].append(trigger)
        for label_text, triggers in trigger_dict.items():
            event_cnt += len(triggers)
            qa_examples.append(
                OrderedDict(
                    [
                        ('text', example['text']),
                        ('query', label2ann[label_text]),
                        ('triggers', triggers)
                    ]
                )
            )
    print("[{}], total {} qa examples, {} events".format(
        data_type, len(qa_examples), event_cnt))
    return qa_examples


def build_matrix(label_map, examples, output_dir):
    label_num = len(label_map)
    adjacency_matrix_cnt = [[0] * label_num for idx in range(label_num)]
    for example in examples:
        label_list = [label['trigger']['label'] for label in example['events']]
        label_list = list(set(label_list))
        for label in label_list:
            label_id = label_map[label]
            adjacency_matrix_cnt[label_id][label_id] += 1
        if len(label_list) < 2:
            continue
        items = list(permutations(label_list, 2))
        for item in items:
            adjacency_matrix_cnt[label_map[item[0]]][label_map[item[1]]] += 1
    adjacency_matrix = [[0.0] * label_num for idx in range(label_num)]
    for label_id in range(label_num):
        label_cnt = adjacency_matrix_cnt[label_id][label_id]
        for label_idx in range(label_num):
            if label_idx == label_id:
                continue
            adjacency_matrix[label_id][label_idx] = adjacency_matrix_cnt[label_id][label_idx] / label_cnt
    adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
    print(adjacency_matrix)
    np.save("{}/gnn_file.npy".format(output_dir), adjacency_matrix)


def split_train_data(examples, radio):
    # random.seed(1)
    random.seed(4)
    example_size = len(examples)
    selected = random.sample(examples, int(example_size * radio))
    print("radio {}({}/{})".format(radio, len(selected), example_size))
    return selected


def test(input_file):
    with open(input_file, 'r', encoding='utf-8') as fr:
        lines = json.load(fr)
        sentences = [' '.join(line['tokens']) for line in lines]
    print(len(sentences))
    print(len(set(sentences)))


def count_data(examples, data_type):
    total_cnt, overlap_cnt, nested_cnt = 0, 0, 0
    for example in examples:
        total_cnt += len(example['events'])
        is_overlap, is_nested = [
            0] * len(example['events']), [0] * len(example['events'])
        span_list = [(event['trigger']['segments']['start_offset'], event['trigger']['segments']['end_offset'], event['trigger']['label'])
                     for event in example['events']]
        for idx, event in enumerate(example['events']):
            trigger = event['trigger']
            cur_span = (trigger['segments']['start_offset'],
                        trigger['segments']['end_offset'], trigger['label'])
            for span_idx, pre_span in enumerate(span_list):
                # if idx == span_idx or cur_span[2] != pre_span[2]:
                #     continue
                if idx == span_idx:
                    continue
                # no overlap & no nested
                if cur_span[1] < pre_span[0] or cur_span[0] > pre_span[1]:
                    continue
                if (cur_span[0] <= pre_span[0] and cur_span[1] >= pre_span[0]) or (cur_span[0] >= pre_span[0] and cur_span[1] <= pre_span[0]):
                    is_nested[idx] = 1
                    is_nested[span_idx] = 1
                else:
                    is_overlap[idx] = 1
                    is_overlap[span_idx] = 1
        overlap_cnt += is_overlap.count(1)
        nested_cnt += is_nested.count(1)
    print("{}, nested: {:4.4f}({}/{}), overlap: {:4.4f}({}/{})".format(data_type, nested_cnt /
                                                                       total_cnt, nested_cnt, total_cnt, overlap_cnt/total_cnt, overlap_cnt, total_cnt))

# def build_few_shot(examples):
#     total_cnt, overlap_cnt, nested_cnt = 0, 0, 0
#     for example in examples:
#         total_cnt += len(example['events'])
#         is_overlap, is_nested = [
#             0] * len(example['events']), [0] * len(example['events'])
#         span_list = [(event['trigger']['segments']['start_offset'], event['trigger']['segments']['end_offset'], event['trigger']['label'])
#                      for event in example['events']]
#         for idx, event in enumerate(example['events']):
#             trigger = event['trigger']
#             cur_span = (trigger['segments']['start_offset'],
#                         trigger['segments']['end_offset'], trigger['label'])
#             for span_idx, pre_span in enumerate(span_list):
#                 if idx == span_idx or cur_span[2] != pre_span[2]:
#                     continue
#                 # no overlap & no nested
#                 if cur_span[1] < pre_span[0] or cur_span[0] > pre_span[1]:
#                     continue
#                 if (cur_span[0] <= pre_span[0] and cur_span[1] >= pre_span[0]) or (cur_span[0] >= pre_span[0] and cur_span[1] <= pre_span[0]):
#                     is_nested[idx] = 1
#                     is_nested[span_idx] = 1
#                 else:
#                     is_overlap[idx] = 1
#                     is_overlap[span_idx] = 1
#         overlap_cnt += is_overlap.count(1)
#         nested_cnt += is_nested.count(1)
#     print("{}, nested: {:4.4f}({}/{}), overlap: {:4.4f}({}/{})".format(data_type, nested_cnt /
#                                                                        total_cnt, nested_cnt, total_cnt, overlap_cnt/total_cnt, overlap_cnt, total_cnt))


def build_few_shot_data(examples, label2id, selected_num=5):
    examples_dict = {idx: [] for idx in range(len(label2id))}
    selected = {}
    for example in examples:
        label_list = [event['trigger']['label'] for event in example['events']]
        label_set = set(label_list)
        for label in label_set:
            examples_dict[label2id[label]].append(example)
    fewshot_selected = []
    for idx, label_name in enumerate(label2id):
        cur_examples = []
        for cur_example in examples_dict[idx]:
            if cur_example['text'] not in selected:
                cur_examples.append(cur_example)
        # print(len(cur_examples))
        s = random.sample(cur_examples, min(selected_num, len(cur_examples)))
        for item in s:
            selected[item['text']] = 0
        fewshot_selected.extend(s)
        print("{}: {}/{}".format(label_name, len(s), len(cur_examples)))
    return fewshot_selected


if __name__ == "__main__":
    # test("train.json")

    # out = load_data("train.json", True)
    # train_examples, train_splited_examples = out['examples'], out['splited_examples']
    # dev_out = load_data(
    #     'dev.json', return_single=True, return_multi=True)
    # dev_examples, dev_single_examples, dev_multi_examples = dev_out[
    #     'examples'], dev_out['single_examples'], dev_out['multi_examples']
    # test_out = load_data(
    #     'test.json', return_single=True, return_multi=True)
    # test_examples, test_single_examples, test_multi_examples = test_out[
    #     'examples'], test_out['single_examples'], test_out['multi_examples']

    # print('total {} triggers, total {} arguments, dump data to dir "processed" and "splited"'.format(
    #     len(trigger_set), len(argument_set)))
    # with open('processed/train.json', 'w', encoding='utf-8') as fw:
    #     for example in train_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('processed/dev.json', 'w', encoding='utf-8') as fw:
    #     for example in dev_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('processed/test.json', 'w', encoding='utf-8') as fw:
    #     for example in test_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    # with open('splited/train.json', 'w', encoding='utf-8') as fw:
    #     for example in train_splited_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('splited/dev.json', 'w', encoding='utf-8') as fw:
    #     for example in dev_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('dev/dev.json', 'w', encoding='utf-8') as fw:
    #     for example in dev_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('dev/dev_single.json', 'w', encoding='utf-8') as fw:
    #     for example in dev_single_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('dev/dev_multi.json', 'w', encoding='utf-8') as fw:
    #     for example in dev_multi_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('splited/test.json', 'w', encoding='utf-8') as fw:
    #     for example in test_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('test/test.json', 'w', encoding='utf-8') as fw:
    #     for example in test_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('test/test_single.json', 'w', encoding='utf-8') as fw:
    #     for example in test_single_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    # with open('test/test_multi.json', 'w', encoding='utf-8') as fw:
    #     for example in test_multi_examples:
    #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    # trigger2id = {trigger: int(idx) for idx,
    #               trigger in enumerate(sorted(trigger_set))}
    # id2trigger = {int(idx): trigger for idx,
    #               trigger in enumerate(sorted(trigger_set))}

    # argument2id = {argument: int(idx) for idx,
    #                argument in enumerate(sorted(argument_set))}
    # id2argument = {int(idx): argument for idx,
    #                argument in enumerate(sorted(argument_set))}

    # with open('processed/trigger_label_map.json', 'w', encoding='utf-8') as fw:
    #     json.dump([id2trigger, trigger2id], fw, indent=4, ensure_ascii=False)

    # with open('processed/argument_label_map.json', 'w', encoding='utf-8') as fw:
    #     json.dump([id2argument, argument2id], fw, indent=4, ensure_ascii=False)

    # with open('splited/trigger_label_map.json', 'w', encoding='utf-8') as fw:
    #     json.dump([id2trigger, trigger2id], fw, indent=4, ensure_ascii=False)

    # with open('splited/argument_label_map.json', 'w', encoding='utf-8') as fw:
    #     json.dump([id2argument, argument2id], fw, indent=4, ensure_ascii=False)

    out = load_data("train.json")
    train_examples = out['examples']
    dev_out = load_data('dev.json')
    dev_examples = dev_out['examples']
    test_out = load_data('test.json')
    test_examples = test_out['examples']

    if not os.path.exists('./processed'):
        os.makedirs('./processed')
    print('total {} triggers, total {} arguments, dump data to dir "processed" and "splited"'.format(
        len(trigger_set), len(argument_set)))
    with open('processed/train.json', 'w', encoding='utf-8') as fw:
        for example in train_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('processed/dev.json', 'w', encoding='utf-8') as fw:
        for example in dev_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('processed/test.json', 'w', encoding='utf-8') as fw:
        for example in test_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    trigger2id = {trigger: int(idx) for idx,
                  trigger in enumerate(sorted(trigger_set))}
    id2trigger = {int(idx): trigger for idx,
                  trigger in enumerate(sorted(trigger_set))}

    argument2id = {argument: int(idx) for idx,
                   argument in enumerate(sorted(argument_set))}
    id2argument = {int(idx): argument for idx,
                   argument in enumerate(sorted(argument_set))}

    with open('trigger_annotation.json', 'r') as fr:
        label2ann = json.load(fr)

    with open('processed/trigger_label_map.json', 'w', encoding='utf-8') as fw:
        json.dump([id2trigger, trigger2id], fw, indent=4, ensure_ascii=False)

    with open('processed/argument_label_map.json', 'w', encoding='utf-8') as fw:
        json.dump([id2argument, argument2id], fw, indent=4, ensure_ascii=False)

    with open('processed/trigger_annotation.txt', 'w', encoding='utf-8') as fw:
        for idx in range(len(id2trigger)):
            trigger = id2trigger[idx]
            fw.write(label2ann[trigger]+'\n')

    if not os.path.exists('./qa_format'):
        os.makedirs('./qa_format')
    qa_train_examples = build_qa_format_data(
        train_examples, label2ann, 'train')
    qa_dev_examples = build_qa_format_data(
        dev_examples, label2ann, 'dev')
    qa_test_examples = build_qa_format_data(
        test_examples, label2ann, 'test')

    with open('qa_format/train.json', 'w', encoding='utf-8') as fw:
        for example in qa_train_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('qa_format/dev.json', 'w', encoding='utf-8') as fw:
        for example in qa_dev_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    with open('qa_format/test.json', 'w', encoding='utf-8') as fw:
        for example in qa_test_examples:
            fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    with open('qa_format/trigger_label_map.json', 'w', encoding='utf-8') as fw:
        json.dump([id2trigger, trigger2id], fw, indent=4, ensure_ascii=False)

    with open('qa_format/argument_label_map.json', 'w', encoding='utf-8') as fw:
        json.dump([id2argument, argument2id], fw, indent=4, ensure_ascii=False)

    build_matrix(trigger2id, train_examples, 'processed')

    count_data(train_examples, 'train')
    count_data(dev_examples, 'dev')
    count_data(test_examples, 'test')
    # for radio in np.arange(0.05, 1.0, 0.05):
    #     selected = split_train_data(train_examples, radio)
    #     with open('processed/train_{:0.2f}.json'.format(radio), 'w', encoding='utf-8') as fw:
    #         for example in selected:
    #             fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    if not os.path.exists('./fewshot'):
        os.makedirs('./fewshot')
    for selected_cnt in [1, 5]:
        random.seed(1)
        few_examples = build_few_shot_data(
            train_examples, trigger2id, selected_num=selected_cnt)
        with open('fewshot/train_{}_shot.json'.format(selected_cnt), 'w', encoding='utf-8') as fw:
            for example in few_examples:
                fw.write(json.dumps(example, ensure_ascii=False) + '\n')
        # few_examples = random.sample(
        #     train_examples, int(len(train_examples)*0.1))
        # with open('fewshot/train_10_shot.json', 'w', encoding='utf-8') as fw:
        #     for example in few_examples:
        #         fw.write(json.dumps(example, ensure_ascii=False) + '\n')

    for radio in [0.01, 0.05]:
        # radio = 0.01
        selected = split_train_data(train_examples, radio)
        with open('fewshot/train_{:0.2f}.json'.format(radio), 'w', encoding='utf-8') as fw:
            for example in selected:
                fw.write(json.dumps(example, ensure_ascii=False) + '\n')
    for radio in np.arange(0.1, 1.0, 0.1):
        selected = split_train_data(train_examples, radio)
        with open('fewshot/train_{:0.2f}.json'.format(radio), 'w', encoding='utf-8') as fw:
            for example in selected:
                fw.write(json.dumps(example, ensure_ascii=False) + '\n')
