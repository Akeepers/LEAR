import argparse

from numpy.lib.function_base import extract
import models as models
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import BertWordPieceTokenizer
from models.model_event import BertTriggerCrf, BertTriggerSoftmax
import torch
import numpy as np
import random
import torch.optim as optim
from transformers import AdamW, BertTokenizer
from callback.lr_scheduler import get_linear_schedule_with_warmup
from utils.common import seed_everything, json_to_text, init_logger, logger, load_model, EventTypeLabel
from utils.losses import *
import json
import jsonlines
import time
from data_loader import collate_fn_trigger_crf, data_processors as processors
from utils.finetuning_args import get_argparse, print_arguments
from utils.evaluate import *
from prefetch_generator import BackgroundGenerator
import logging
import multiprocessing
import pickle
from tqdm import tqdm, trange
from collections import OrderedDict

MODEL_CLASSES = {
    'crf_trigger': (processors['crf_trigger'], collate_fn_trigger_crf),
    'softmax_trigger': (processors['softmax_trigger'], collate_fn_trigger_crf)
}



class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_label_data(args, tokenizer):
    with open(args.label_str_file, 'r', encoding='utf-8') as fr:
        label_str_list = [line.strip() for line in fr.readlines()]
    token_ids, input_mask, max_len = [], [], 0
    for label_str in label_str_list:
        # sub_tokens = [tokenizer.cls_token] + \
        #     tokenizer.tokenize(label_str) + [tokenizer.sep_token]
        # token_id = tokenizer.convert_tokens_to_ids(sub_tokens)
        # input_mask.append([1] * len(token_id))

        encoded_results = tokenizer.encode(label_str, add_special_tokens=True)
        token_id = encoded_results.ids
        input_mask.append(encoded_results.attention_mask)
        max_len = max(max_len, len(token_id))
        token_ids.append(token_id)
    assert max_len <= args.max_seq_length and len(
        token_ids) == args.first_label_num
    for idx in range(len(token_ids)):
        # print(token_ids[idx])
        # print(input_mask[idx])
        padding_length = max_len - len(token_ids[idx])
        token_ids[idx] += [0] * padding_length
        input_mask[idx] += [0] * padding_length

    token_ids = np.array(token_ids)
    input_mask = np.array(input_mask)
    token_type_ids = np.zeros_like(token_ids)
    # token_type_ids = np.ones_like(token_ids)
    return token_ids, input_mask, token_type_ids


def load_and_cache_examples(args, tokenizer, processor, input_file, data_type='train', return_second_label_dict=True, dump_vocab=True):
    if os.path.exists(args.model_name_or_path):
        pretrain_model_name = str(args.model_name_or_path).split('/')[-1]
    else:
        pretrain_model_name = str(args.model_name_or_path)
    data_prefix = "".join(input_file.split("/")[-1].split(".")[:-1])
    temp = 'crf' if args.model_name == "trigger_extraction_crf" else "softmax"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, '{}_cached_{}_{}_{}_{}_{}_{}'.format(
        temp,
        pretrain_model_name,
        args.data_name,
        # data_type,
        data_prefix,
        str(args.task_type),
        ('uncased' if args.do_lower_case else 'cased'),
        str(args.max_seq_length)))
    cached_second_label_dict_file = '{}_second_label_dict'.format(
        cached_features_file)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file '{}'".format(
            cached_features_file))
        with open(cached_features_file, 'rb') as fr:
            features = pickle.load(fr)
            logger.info("total records: {}".format(len(features)))

        if data_type != "train" and return_second_label_dict:
            assert os.path.exists(cached_second_label_dict_file)
            logger.info("Loading second label dictionary from cached file '{}'".format(
                cached_second_label_dict_file))
            with open(cached_second_label_dict_file, 'rb') as fr:
                second_label_dict = pickle.load(fr)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples(input_file)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(input_file)
        else:
            examples = processor.get_test_examples(input_file)
        out = processor.convert_examples_to_feature(examples=examples,
                                                    tokenizer=tokenizer,
                                                    first_label_map_file=args.first_label_file,
                                                    second_label_map_file=args.second_label_file,
                                                    max_seq_len=args.max_seq_length,
                                                    data_type=data_type,
                                                    pad_id=0,
                                                    do_lower_case=args.do_lower_case,
                                                    add_event_type=args.add_event_type,
                                                    is_maven=args.is_maven
                                                    )
        features = out['features']
        logger.info("Saving features into cached file '{}', total_records: {}".format(
            cached_features_file, len(features)))
        logger.info(out['stat_info'])
        with open(cached_features_file, 'wb') as fw:
            pickle.dump(features, fw)
        if data_type != "train" and return_second_label_dict:
            second_label_dict = out['second_label_dict']
            logger.info("Saving second_label_dict into cached file '{}'".format(
                cached_second_label_dict_file))
            with open(cached_second_label_dict_file, 'wb') as fw:
                pickle.dump(second_label_dict, fw)
    if data_type != "train" and return_second_label_dict:
        return features, second_label_dict
    return features


def evaluate_trigger(args, model, tokenizer, processor, input_file, output=False, output_eval_info=False, data_type='dev', joiner_symbol=" "):
    dataset = load_and_cache_examples(
        args, tokenizer, processor, data_type=data_type, input_file=input_file, return_second_label_dict=False)
    args.eval_batch_size = 1
    dev_dataloader = DataLoader(dataset=dataset,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=multiprocessing.cpu_count()//4,
                                collate_fn=MODEL_CLASSES[args.task_type][1])

    if output:
        output_results = []
        order = ["label", "text"]
        id_list = []
        doc_results = {}
    id2label = processor.get_labels(args.first_label_file)[0]
    trigger_correct_num, trigger_infer_num, trigger_gold_num = 0, 0, 0
    event_type_label_num, event_type_infer_num, event_type_correct_num = 0, 0, 0

    model.eval()

    if args.model_name == "trigger_extraction_with_label_siamese":
        label_token_ids, label_input_mask, label_token_type_ids = get_label_data(
            args, tokenizer)
        batch_label_token_ids = torch.tensor(
            label_token_ids, dtype=torch.long).to(args.device)
        batch_label_input_mask = torch.tensor(
            label_input_mask, dtype=torch.float32).to(args.device)
        batch_label_token_type_ids = torch.tensor(
            label_token_type_ids, dtype=torch.long).to(args.device)
    # for step, data in enumerate(dev_dataloader):
    add_label_info = True
    eval_start_time = time.time()
    with tqdm(dev_dataloader, desc="Evaluation") as eval_bar:
        for step, data in enumerate(eval_bar):
            with torch.no_grad():
                # if args.model_name == "trigger_extraction_with_label_siamese":
                #     data['label_token_ids'] = batch_label_token_ids
                #     data['label_input_mask'] = batch_label_input_mask
                #     data['label_token_type_ids'] = batch_label_token_type_ids
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'ori_tokens']:
                        data[key] = data[key].to(args.device)

                loss, logits = model(data)
                # batch_gold_labels = data["golden_label"]

                batch_position_dict = []
                for seq_token_ids in data['token_ids'].cpu().numpy():
                    # tokens = tokenizer.convert_ids_to_tokens(seq_token_ids)
                    tokens = [tokenizer.id_to_token(
                        token_id) for token_id in seq_token_ids]
                    position_dict, _, _ = restore_subtoken(
                        tokens, return_tokens=False)
                    batch_position_dict.append(position_dict)
                # print(logits.shape)
                # print(data['token_ids'].shape)
                if args.model_name == "trigger_extraction_crf":
                    batch_trigger_spans = model.crf.decode(
                        logits, data['input_mask']).squeeze(0).cpu().numpy().tolist()
                else:
                    batch_trigger_spans = np.argmax(
                        logits.cpu().numpy(), axis=2).tolist()
                for batch_idx, (seq_trigger_gold_labels, seq_trigger_infer_labels) in enumerate(zip(data["golden_label"], batch_trigger_spans)):
                    # cur_trigger_gold_labels = set(seq_trigger_gold_labels)

                    # cur_trigger_labels = set(get_span_crf(seq_trigger_infer_labels, id2label))
                    # trigger_gold_num += len(cur_trigger_gold_labels)
                    # trigger_infer_num += len(cur_trigger_labels)
                    # trigger_correct_num += len(cur_trigger_gold_labels &
                    #                            cur_trigger_labels)

                    cur_trigger_labels_ = get_span_crf(
                        seq_trigger_infer_labels, id2label)
                    cur_trigger_labels, seq_trigger_gold_labels_ = [], []
                    for infer_label in cur_trigger_labels_:
                        start_idx = batch_position_dict[batch_idx][infer_label.start_idx]
                        end_idx = batch_position_dict[batch_idx][infer_label.end_idx]
                        cur_trigger_labels.append(
                            LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=infer_label.label_id))
                    cur_trigger_labels = set(cur_trigger_labels)
                    for infer_label in seq_trigger_gold_labels:
                        start_idx = batch_position_dict[batch_idx][infer_label.start_idx]
                        end_idx = batch_position_dict[batch_idx][infer_label.end_idx]
                        seq_trigger_gold_labels_.append(
                            LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=infer_label.label_id))
                    seq_trigger_gold_labels_ = set(seq_trigger_gold_labels_)
                    trigger_gold_num += len(seq_trigger_gold_labels_)
                    trigger_infer_num += len(cur_trigger_labels)
                    trigger_correct_num += len(seq_trigger_gold_labels_ &
                                               cur_trigger_labels)
                    if output:
                        seq_token_ids = data['token_ids'][batch_idx].cpu(
                        ).numpy()
                        if args.is_maven and data_type == 'test':
                            infer_label_dict = {(label.start_idx, label.end_idx): label.label_id
                                                for label in cur_trigger_labels
                                                }
                            golden_label_dict = {(label.start_idx, label.end_idx): label.label_id
                                                 for label in cur_trigger_gold_labels
                                                 }
                            cur_infer_labels = set([(label.start_idx, label.end_idx)
                                                    for label in cur_trigger_labels])
                            cur_golden_labels = set([(label.start_idx, label.end_idx)
                                                     for label in cur_trigger_gold_labels])
                            inner_labels = cur_infer_labels & cur_golden_labels
                            sub_labels = cur_golden_labels - inner_labels
                            label_results = []
                            for label in inner_labels:
                                label_results.append(
                                    {
                                        'id': golden_label_dict[label],
                                        'type_id': infer_label_dict[label] + 1
                                    }
                                )
                            for label in sub_labels:
                                label_results.append(
                                    {
                                        'id': golden_label_dict[label],
                                        'type_id': 0
                                    }
                                )
                            doc_id = data['ids'][batch_idx]['doc_id']
                            if doc_id in doc_results:
                                doc_results[doc_id].extend(label_results)
                            else:
                                doc_results[doc_id] = label_results
                        else:
                            infer_labels = [EventTypeLabel(text=tokenizer.decode(
                                seq_token_ids[infer_span.start_idx:infer_span.end_idx+1]), label=id2label[infer_span.label_id]) for infer_span in cur_trigger_labels]
                            golden_labels = [EventTypeLabel(text=tokenizer.decode(
                                seq_token_ids[golden_span.start_idx:golden_span.end_idx+1]), label=id2label[golden_span.label_id]) for golden_span in cur_trigger_gold_labels]

                            infer_labels = set(infer_labels)
                            golden_labels = set(golden_labels)
                            output_results.append(
                                json.dumps({
                                    'text': tokenizer.decode(seq_token_ids, skip_special_tokens=True),
                                    'event_list_golden': [
                                        dict(zip(order, label)) for label in golden_labels
                                    ],
                                    'event_list_infer': [
                                        dict(zip(order, label)) for label in infer_labels
                                    ],
                                    'new': [
                                        dict(zip(order, label)) for label in infer_labels - golden_labels
                                    ],
                                    'lack': [
                                        dict(zip(order, label)) for label in golden_labels - infer_labels
                                    ]
                                }, ensure_ascii=False)
                            )
                if args.use_auxiliary_task:

                    for infer, golden in zip(infer_trigger_type.cpu().numpy(), data['first_types']):
                        for infer_label, golden_label in zip(infer, golden):
                            if infer_label > 0.5:
                                event_type_infer_num += 1
                            if int(golden_label) == 1:
                                event_type_label_num += 1
                            if infer_label > 0.5 and int(golden_label) == 1:
                                event_type_correct_num += 1

                if output:
                    for batch_idx in range(len(data['ids'])):
                        id_list.append(data['ids'][batch_idx])
                        output_results[data['ids'][batch_idx]] = {
                            'text': joiner_symbol.join(data['ori_tokens'][batch_idx][1:-1]),
                            'events_golden': batch_gold_labels[batch_idx],
                            'events_infer': []
                        }
    logger.info('eval {}, total {:5.2f}s'.format(
        data_type, time.time() - eval_start_time))
    if output:
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        path = os.path.join(
            args.result_dir, "{}_result_trigger.json".format(data_type))
        if args.is_maven and data_type == 'test':
            temp = [{
                'id': key,
                'predictions': value
            }for key, value in doc_results.items()]
            with jsonlines.open(path+'l', 'w') as fw:
                for item in temp:
                    fw.write(item)

        else:
            with open(path, 'w', encoding='utf-8') as fw:
                for item in output_results:
                    fw.write(item+'\n')

        # data_prefix = input_file.split('/')[-1].split('.')[0]
        # with open("{}/{}_golden.json".format(args.result_dir, data_prefix), 'w', encoding='utf-8') as fw:
        #     for example_id in id_list:
        #         d = OrderedDict([
        #             ('id', example_id),
        #             ('text', output_results[example_id]['text']),
        #             ('events', output_results[example_id]['events_golden'])])
        #         fw.write(json.dumps(d, indent=4, ensure_ascii=False)+'\n')
        # with open("{}/{}_infer.json".format(args.result_dir, data_prefix), 'w', encoding='utf-8') as fw:
        #     for example_id in id_list:
        #         d = OrderedDict([
        #             ('id', example_id),
        #             ('text', output_results[example_id]['text']),
        #             ('events', output_results[example_id]['events_infer'])
        #         ])
        #         fw.write(json.dumps(d, indent=4, ensure_ascii=False) + '\n')

    result_dict = {
        'trigger_golden_num': trigger_gold_num,
        'trigger_infer_num': trigger_infer_num,
        'trigger_correct_num': trigger_correct_num,
        'event_golden_num': event_type_label_num,
        'event_infer_num': event_type_infer_num,
        'event_correct_num': event_type_correct_num,
    }
    result_dict['trigger_precision'], result_dict['trigger_recall'], result_dict['trigger_f1'] = calculate_f1(
        label_num=trigger_gold_num, infer_num=trigger_infer_num, correct_num=trigger_correct_num)
    result_dict['event_precision'], result_dict['event_recall'], result_dict['event_f1'] = calculate_f1(
        label_num=event_type_label_num, infer_num=event_type_infer_num, correct_num=event_type_correct_num)
    if args.use_auxiliary_task:
        result_dict['f1'] = (result_dict['trigger_f1'] +
                             result_dict['event_f1']) / 2
    else:
        result_dict['f1'] = result_dict['trigger_f1']
    if output_eval_info:
        data_prefix = input_file.split('/')[-1].split('.')[0]
        logger.info("***** Eval results: {} *****".format(data_prefix))
        if args.use_auxiliary_task:
            logger.info(
                "trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}); event f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".
                format(result_dict["trigger_f1"], result_dict["trigger_precision"],
                       result_dict["trigger_correct_num"], result_dict["trigger_infer_num"], result_dict[
                    "trigger_recall"], result_dict["trigger_correct_num"], result_dict["trigger_golden_num"],
                    result_dict["event_f1"], result_dict["event_precision"],
                    result_dict["event_correct_num"], result_dict["event_infer_num"], result_dict["event_recall"], result_dict["event_correct_num"], result_dict["event_golden_num"]))
        else:
            logger.info(
                "trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".
                format(result_dict["trigger_f1"], result_dict["trigger_precision"],
                       result_dict["trigger_correct_num"], result_dict["trigger_infer_num"], result_dict[
                    "trigger_recall"], result_dict["trigger_correct_num"], result_dict["trigger_golden_num"]))
    return result_dict


def train(args, model, tokenizer, processor):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.eval_test = False if args.is_maven else args.eval_test
    train_dataset = load_and_cache_examples(
        args, tokenizer=tokenizer, processor=processor, data_type="train", input_file=args.train_set, return_second_label_dict=False)
    train_dataloader = DataLoaderX(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   #   pin_memory=True,
                                   drop_last=True,
                                   num_workers=multiprocessing.cpu_count()//4,
                                   collate_fn=MODEL_CLASSES[args.task_type][1])

    t_total = len(
        train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.val_step = max(1, len(train_dataloader) // args.eval_per_epoch)
    # print("{}, {}, {}".format(len(train_dataloader),
    #                           args.eval_per_epoch, args.val_step))
    # define the optimizer

    bert_parameters = model.bert.named_parameters()
    classifier_params = model.classifier.named_parameters()

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate},

        {"params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in classifier_params if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},


        # {"params": [p for n, p in first_type_start_params if not any(nd in n for nd in no_decay)],
        #  "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        # {"params": [p for n, p in first_type_start_params if any(
        #     nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
        # {"params": [p for n, p in first_type_end_params if not any(nd in n for nd in no_decay)],
        #  "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        # {"params": [p for n, p in first_type_end_params if any(
        #     nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
    ]

    if args.model_name == "trigger_extraction_crf":
        crf_params = model.crf.named_parameters()
        optimizer_grouped_parameters += [{"params": [p for n, p in crf_params if not any(nd in n for nd in no_decay)],
                                          "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
                                         {"params": [p for n, p in crf_params if any(
                                             nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr}, ]

    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # check the output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Total warmup steps = %d", warmup_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  lr of encoder = {}, lr of task_layer = {}".format(
        args.learning_rate, args.learning_rate * args.task_layer_lr))
    logger.info('\n')

    model.zero_grad()
    # Added here for reproductibility (even between python 2 and 3)
    seed_everything(args.seed)

    if args.model_name == "trigger_extraction_with_label_siamese":
        label_token_ids, label_input_mask, label_token_type_ids = get_label_data(
            args, tokenizer)
        batch_label_token_ids = torch.tensor(
            label_token_ids, dtype=torch.long).to(args.device)
        batch_label_input_mask = torch.tensor(
            label_input_mask, dtype=torch.float32).to(args.device)
        batch_label_token_type_ids = torch.tensor(
            label_token_type_ids, dtype=torch.long).to(args.device)
    if args.do_train:
        model.train()
        global_step = 0
        loss_sum = 0

        best_result = {'f1': 0.0}
        info_str = ""

        best_step = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        add_label_info = True
        for epoch in range(int(args.num_train_epochs)):
            train_bar = tqdm(train_dataloader, desc="Training")
            one_epoch_start_time = time.time()
            for data in train_bar:
                global_step += 1
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'ori_tokens']:
                        data[key] = data[key].to(args.device)

                total_loss, logits = model(data)

                if args.n_gpu > 1:
                    total_loss = total_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

                total_loss.backward()
                writer.add_scalar('loss/train', total_loss.item(), global_step)

                train_bar.set_description(
                    "epoch:{}/{} - step:{}, loss:{:.6f} ".format(epoch + 1, int(args.num_train_epochs), global_step, total_loss.item() * args.gradient_accumulation_steps))

                add_label_info = False
                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    add_label_info = True

                if args.do_eval and global_step > args.val_skip_step and global_step % args.val_step == 0:
                    eval_start_time = time.time()

                    test_model = model.module if isinstance(
                        model, torch.nn.DataParallel) else model
                    # call the test function

                    eval_result = evaluate_trigger(
                        args, test_model, tokenizer, processor, args.dev_set)
                    if 'cuda' in str(args.device):
                        torch.cuda.empty_cache()
                    writer.add_scalar(
                        "trigger_f1/dev", eval_result["trigger_f1"], global_step)
                    writer.add_scalar(
                        "trigger_precision/dev", eval_result["trigger_precision"], global_step)
                    writer.add_scalar(
                        "trigger_recall/dev", eval_result["trigger_recall"], global_step)
                    if args.use_auxiliary_task:
                        writer.add_scalar(
                            "event_f1/dev", eval_result["event_f1"], global_step)
                        writer.add_scalar(
                            "event_precision/dev", eval_result["event_precision"], global_step)
                        writer.add_scalar(
                            "event_recall/dev", eval_result["event_recall"], global_step)

                    if eval_result["f1"] > best_result["f1"]:
                        best_result.update(eval_result)
                        best_result["step"] = global_step
                        best_result["epoch"] = epoch + 1
                        # save the best model
                        output_dir = args.output_dir
                        model_to_save = model.module if isinstance(
                            model, torch.nn.DataParallel) else model
                        torch.save(model_to_save.state_dict(),
                                   os.path.join(output_dir, "model.bin"))
                        torch.save(scheduler.state_dict(), os.path.join(
                            output_dir, "scheduler.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(
                            output_dir, "optimizer.pt"))
                    logger.info("**Dev**")
                    logger.info(eval_result)
                    if eval_result["f1"] > 0:
                        if args.use_auxiliary_task:
                            logger.info(
                                "best model: step {}, epoch: {} , f1: {:4.4f} -- trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}); event f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".
                                format(best_result["step"], best_result['epoch'], best_result["f1"], best_result["trigger_f1"], best_result["trigger_precision"],
                                       best_result["trigger_correct_num"], best_result["trigger_infer_num"], best_result[
                                    "trigger_recall"], best_result["trigger_correct_num"], best_result["trigger_golden_num"],
                                    best_result["event_f1"], best_result["event_precision"],
                                    best_result["event_correct_num"], best_result["event_infer_num"], best_result["event_recall"], best_result["event_correct_num"], best_result["event_golden_num"]))
                        else:
                            logger.info(
                                "best model: step {}, epoch: {} , f1: {:4.4f} -- trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{});".
                                format(best_result["step"], best_result['epoch'], best_result["f1"], best_result["trigger_f1"], best_result["trigger_precision"],
                                       best_result["trigger_correct_num"], best_result["trigger_infer_num"], best_result[
                                    "trigger_recall"], best_result["trigger_correct_num"], best_result["trigger_golden_num"]))
                    if args.eval_test:
                        eval_result = evaluate_trigger(
                            args, test_model, tokenizer, processor, args.test_set, data_type='test')
                        if 'cuda' in str(args.device):
                            torch.cuda.empty_cache()
                        writer.add_scalar(
                            "trigger_f1/test", eval_result["trigger_f1"], global_step)
                        writer.add_scalar(
                            "trigger_precision/test", eval_result["trigger_precision"], global_step)
                        writer.add_scalar(
                            "trigger_recall/test", eval_result["trigger_recall"], global_step)
                        if args.use_auxiliary_task:
                            writer.add_scalar(
                                "event_f1/test", eval_result["event_f1"], global_step)
                            writer.add_scalar(
                                "event_precision/test", eval_result["event_precision"], global_step)
                            writer.add_scalar(
                                "event_recall/test", eval_result["event_recall"], global_step)
                        logger.info("**Test**")
                        logger.info(eval_result)
                    model.train()

                    # manually release the unused cache
                    if 'cuda' in str(args.device):
                        torch.cuda.empty_cache()

            logger.info("one epoch end, total {:5.2f}s".format(
                time.time() - one_epoch_start_time))

        logger.info("finish training")
        if args.task_type == 'relation':
            logger.info(
                "best model: step {}, best f1: {:4.4f}, precision: {:4.4f}({}/{}), recall: {:4.4f}({}/{}), total time: {:5.2f}s".
                format(best_result["step"], best_result["f1"], best_result["precision"],
                       best_result["correct_num"], best_result["infer_num"], best_result[
                    "recall"], best_result["correct_num"], best_result["total_num"],
                    time.time() - init_time))
        else:
            if args.use_auxiliary_task:
                logger.info("best model: step {}, f1: {:4.4f} -- trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}); event f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                            format(best_result["step"], best_result["f1"], best_result["trigger_f1"], best_result["trigger_precision"],
                                   best_result["trigger_correct_num"], best_result["trigger_infer_num"], best_result[
                                "trigger_recall"], best_result["trigger_correct_num"], best_result["trigger_golden_num"],
                                best_result["event_f1"], best_result["event_precision"],
                                best_result["event_correct_num"], best_result["event_infer_num"], best_result["event_recall"], best_result["event_correct_num"], best_result["event_golden_num"], time.time() - init_time))
            else:
                logger.info(
                    "best model: step {}, epoch: {} , f1: {:4.4f} -- trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{});".
                    format(best_result["step"], best_result['epoch'], best_result["f1"], best_result["trigger_f1"], best_result["trigger_precision"],
                           best_result["trigger_correct_num"], best_result["trigger_infer_num"], best_result[
                        "trigger_recall"], best_result["trigger_correct_num"], best_result["trigger_golden_num"]))


def main():
    args = get_argparse().parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if args.model_name_or_path.endswith('-uncased') and (not args.do_lower_case):
        raise ValueError(
            "use uncased model, 'do_lower_case' must be True")
    if args.model_name_or_path.endswith('-cased') and args.do_lower_case:
        raise ValueError(
            "use cased model, 'do_lower_case' must be False")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, 'logs'))
        os.mkdir(os.path.join(args.output_dir, 'tensorboard'))
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(
        log_file=args.output_dir + f'/logs/{args.task_type}-{time_}.log')
    args.device = torch.device("cuda" if torch.cuda.is_available()
                               and args.use_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    processor = MODEL_CLASSES[args.task_type][0]()
    args.first_label_num = len(processor.get_labels(args.first_label_file)[0])
    # Set seed
    seed_everything(args.seed)

    # init_model
    if args.model_name == "trigger_extraction_crf":
        model = BertTriggerCrf(args)
    elif args.model_name == "trigger_extraction_softmax":
        model = BertTriggerSoftmax(args)
    model.to(args.device)

    # tokenizer = BertTokenizer.from_pretrained(
    #     args.model_name_or_path, do_lower_case=args.do_lower_case)

    tokenizer = BertWordPieceTokenizer(
        vocab=args.vocab_file, lowercase=args.do_lower_case)

    # logger.info("Training/evaluation parameters %s", args)
    print_arguments(args, logger)
    # print_arguments(args, logger)
    if args.do_train:
        train(args, model, tokenizer, processor)
    if args.do_eval:
        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        load_model(model, args.output_dir, 'model.bin')

        # args.dev_set = '/data/yangpan/workspace/onepass_ie/data/ace05/dev'
        # args.test_set = '/data/yangpan/workspace/onepass_ie/data/ace05/test'

        if args.dev_set is not None:
            if os.path.isdir(args.dev_set):
                for dev_file in os.listdir(args.dev_set):
                    evaluate_trigger(args, model, tokenizer, processor,
                                     output=False, output_eval_info=True, data_type='dev', input_file=os.path.join(args.dev_set, dev_file))
            else:
                evaluate_trigger(args, model, tokenizer, processor,
                                 output=args.dump_result, output_eval_info=True, data_type='dev', input_file=args.dev_set)
        if args.test_set is not None:
            if os.path.isdir(args.test_set):
                for test_file in os.listdir(args.test_set):
                    evaluate_trigger(args, model, tokenizer, processor,
                                     output=False, output_eval_info=True, data_type='test', input_file=os.path.join(args.test_set, test_file))
            else:
                evaluate_trigger(args, model, tokenizer, processor,
                                 output=args.dump_result, output_eval_info=True, data_type='test', input_file=args.test_set)


if __name__ == "__main__":
    main()
