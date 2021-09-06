import argparse
import models as models
import os
import sys
import shutil
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import BertWordPieceTokenizer
from models.model_ner import BertNerCrf, BertNerSoftmax
# from models.model_ner import BertNerSpan
import torch
import numpy as np
from utils.losses import *
import random
import torch.optim as optim
from transformers import AdamW, BertTokenizer
from callback.lr_scheduler import get_linear_schedule_with_warmup
from utils.common import EntityLabelWithScore, seed_everything, json_to_text, init_logger, logger, load_model, EntityLabel
import json
import time
from data_loader import RelationDataset, collate_fn_ner_crf, data_processors as processors
from utils.finetuning_args import get_argparse, print_arguments
from utils.evaluate import *
import logging
import multiprocessing
import pickle
from tqdm import tqdm, trange
from prefetch_generator import BackgroundGenerator
from collections import OrderedDict
import faulthandler

MODEL_CLASSES = {
    'ner_crf': (processors['ner_crf'], collate_fn_ner_crf),
    'ner_softmax': (processors['ner_softmax'], collate_fn_ner_crf)
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
    return token_ids, input_mask, token_type_ids


def load_and_cache_examples(args, tokenizer, processor, input_file, data_type='train'):
    if os.path.exists(args.model_name_or_path):
        pretrain_model_name = str(args.model_name_or_path).split('/')[-1]
    else:
        pretrain_model_name = str(args.model_name_or_path)
    # data_prefix = input_file.split("/")[-1].split(".")[0]
    data_prefix = "".join(input_file.split("/")[-1].split(".")[:-1])
    temp = 'crf' if args.model_name == "ner_crf" else "softmax"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
        pretrain_model_name,
        args.data_name,
        # data_type,
        data_prefix,
        str(args.task_type),
        ('uncased' if args.do_lower_case else 'cased'),
        str(args.max_seq_length if data_type == 'train' else args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'rb') as fr:
            # features = torch.load(cached_features_file)
            features = pickle.load(fr)
            logger.info("total records: {}".format(len(features)))
            # print(type(features[0]))

    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # id2label, label2id = processor.get_labels(args.second_label_file)
        if data_type == 'train':
            examples = processor.get_train_examples(input_file)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(input_file)
        else:
            examples = processor.get_test_examples(input_file)
        features, stat_info = processor.convert_examples_to_feature(examples=examples,
                                                                    tokenizer=tokenizer,
                                                                    first_label_map_file=args.first_label_file,
                                                                    second_label_map_file=args.second_label_file,
                                                                    max_seq_len=args.max_seq_length,
                                                                    data_type=data_type,
                                                                    pad_id=0,
                                                                    do_lower_case=args.do_lower_case,
                                                                    is_chinese=args.is_chinese,
                                                                    )
        logger.info("Saving features into cached file {}, total_records: {}, {}".format(
                    cached_features_file, len(features), stat_info))
        # torch.save(features, cached_features_file)
        with open(cached_features_file, 'wb') as fw:
            pickle.dump(features, fw)
    return features


def evaluate(args, model, tokenizer, processor, input_file, output=False, output_eval_info=False, data_type='dev', joiner_symbol=" "):
    dev_dataset = load_and_cache_examples(
        args, tokenizer, processor, input_file, data_type=data_type)
    args.eval_batch_size = 1
    dev_dataloader = DataLoaderX(dataset=dev_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=multiprocessing.cpu_count()//4,
                                 collate_fn=MODEL_CLASSES[args.task_type][1])

    if output:
        output_results = []
        order = ["label", "text"]
    id2label = processor.get_labels(args.first_label_file)[0]
    correct_num, infer_num, gold_num = 0, 0, 0
    type_label_num, type_infer_num, type_correct_num = 0, 0, 0

    if args.model_name == 'bert_ner_with_label_siamese' or args.model_name == 'bert_ner_matrix_with_label_siamese':
        label_token_ids, label_input_mask, label_token_type_ids = get_label_data(
            args, tokenizer)
        batch_label_token_ids = torch.tensor(
            label_token_ids, dtype=torch.long).to(args.device)
        batch_label_input_mask = torch.tensor(
            label_input_mask, dtype=torch.float32).to(args.device)
        batch_label_token_type_ids = torch.tensor(
            label_token_type_ids, dtype=torch.long).to(args.device)
        # batch_label_token_ids = torch.tensor(
        #     label_token_ids, dtype=torch.long)
        # batch_label_input_mask = torch.tensor(
        #     label_input_mask, dtype=torch.float32)
        # batch_label_token_type_ids = torch.tensor(
        #     label_token_type_ids, dtype=torch.long)
    model.eval()
    eval_start_time = time.time()
    dev_bar = tqdm(dev_dataloader, desc="Evaluation")
    for data in dev_bar:
        # for step, data in enumerate(dev_dataloader):
        with torch.no_grad():
            for key in data.keys():
                if key not in ['golden_label', 'ids']:
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

            if args.model_name == "bert_ner_crf":
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
                gold_num += len(seq_trigger_gold_labels_)
                infer_num += len(cur_trigger_labels)
                correct_num += len(seq_trigger_gold_labels_ &
                                   cur_trigger_labels)
    print("eval {}, total {:5.2f}s".format(
        data_type, time.time()-eval_start_time))
    if output:
        # check the result dir
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        path = os.path.join(
            args.result_dir, "{}_result_ner.json".format(data_type))
        with open(path, 'w', encoding='utf-8') as fw:
            for item in output_results:
                fw.write(item+'\n')

    result_dict = {
        'golden_num': gold_num,
        'infer_num': infer_num,
        'correct_num': correct_num,
    }

    result_dict['ner_precision'], result_dict['ner_recall'], result_dict['ner_f1'] = calculate_f1(
        label_num=gold_num, infer_num=infer_num, correct_num=correct_num)
    if args.use_auxiliary_task:
        result_dict['type_golden_num'], result_dict['type_infer_num'], result_dict[
            'type_correct_num'] = type_label_num, type_infer_num, type_correct_num
        result_dict['type_precision'], result_dict['type_recall'], result_dict['type_f1'] = calculate_f1(
            label_num=type_label_num, infer_num=type_infer_num, correct_num=type_correct_num)
        result_dict['f1'] = (result_dict['ner_f1'] +
                             result_dict['type_f1']) / 2
    else:
        result_dict['f1'] = result_dict['ner_f1']
    if output_eval_info:
        data_prefix = input_file.split('/')[-1].split('.')[0]
        logger.info("***** Eval results: {} *****".format(data_prefix))
        if args.use_auxiliary_task:
            logger.info(
                "f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}); detection f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(
                    result_dict["ner_f1"], result_dict["ner_precision"], result_dict["correct_num"], result_dict["infer_num"], result_dict[
                        "ner_recall"], result_dict["correct_num"], result_dict["golden_num"],  result_dict["type_f1"], result_dict["type_precision"], result_dict["type_correct_num"], result_dict["type_infer_num"], result_dict[
                        "type_recall"], result_dict["type_correct_num"], result_dict["type_golden_num"]))
        else:
            logger.info(
                "f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(result_dict["ner_f1"], result_dict["ner_precision"],
                                                                           result_dict["correct_num"], result_dict["infer_num"], result_dict[
                    "ner_recall"], result_dict["correct_num"], result_dict["golden_num"]))
    else:
        return result_dict


def train(args, model, tokenizer, processor):

    weight_sum = args.weight_start_loss + \
        args.weight_end_loss + args.weight_span_loss
    args.weight_start_loss = args.weight_start_loss / weight_sum
    args.weight_end_loss = args.weight_end_loss / weight_sum
    args.weight_span_loss = args.weight_span_loss / weight_sum

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(
        args, tokenizer=tokenizer, data_type="train", processor=processor, input_file=args.train_set)
    train_dataloader = DataLoaderX(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   #   pin_memory=True,
                                   drop_last=True,
                                   #    num_workers=multiprocessing.cpu_count()//4,
                                   num_workers=1,
                                   collate_fn=MODEL_CLASSES[args.task_type][1])

    t_total = len(
        train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.val_step = max(1, len(train_dataloader) // args.eval_per_epoch)
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
    ]

    if args.model_name == 'bert_ner_crf':
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

    # optimizer = optim.Adam(filter(
    #     lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

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
    # logger.info("  Total warmup steps = %d", warmup_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  lr of encoder = {}, lr of task_layer = {}".format(
        args.learning_rate, args.learning_rate * args.task_layer_lr))
    logger.info('\n')

    model.zero_grad()
    # Added here for reproductibility (even between python 2 and 3)
    seed_everything(args.seed)

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
        step_list = [idx for idx in range(len(train_dataloader))]
        for epoch in range(int(args.num_train_epochs)):
            train_bar = tqdm(train_dataloader, desc="Training")
            one_epoch_steps = len(train_bar)
            selected = random.sample(
                step_list, int(len(train_dataloader) * 0.1))
            one_epoch_start_time = time.time()
            for cur_step, data in enumerate(train_bar):
                global_step += 1

                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'ori_tokens']:
                        data[key] = data[key].to(args.device)
                total_loss, logits = model(data)

                writer.add_scalar(
                    'loss/train', total_loss.item(), global_step)

                if args.n_gpu > 1:
                    total_loss = total_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

                total_loss.backward()

                train_bar.set_description(
                    "{}/{} step:{}, loss:{:.6}".format(epoch + 1, int(args.num_train_epochs), global_step, total_loss.item()))

                add_label_info = False
                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    # add_label_info = True
                    # print(global_step)

                if args.do_eval and global_step > args.val_skip_step and global_step % args.val_step == 0:
                    if 'cuda' in str(args.device):
                        torch.cuda.empty_cache()
                    eval_start_time = time.time()
                    model.eval()
                    if isinstance(model, torch.nn.DataParallel):
                        test_model = model.module
                    else:
                        test_model = model
                    # call the test function
                    eval_result = evaluate(
                        args, test_model, tokenizer, processor=processor, input_file=args.dev_set, joiner_symbol="")
                    writer.add_scalar(
                        "f1/dev", eval_result["f1"], global_step)
                    if args.use_auxiliary_task:
                        writer.add_scalar(
                            "ner_f1/dev", eval_result["ner_f1"], global_step)
                        writer.add_scalar(
                            "detection_f1/dev", eval_result["type_f1"], global_step)
                    logger.info("[dev], f1: {}\n".format(eval_result['f1']))
                    if eval_result["f1"] > best_result["f1"]:
                        best_result.update(eval_result)
                        best_result["step"] = global_step
                        best_result['epoch'] = epoch+1
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

                    if eval_result["f1"] > 0:
                        if args.use_auxiliary_task:
                            logger.info(
                                "best model: epoch {}, step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}); detection f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(
                                    best_result["epoch"], best_result["step"],
                                    best_result["ner_f1"], best_result["ner_precision"],
                                    best_result["correct_num"], best_result["infer_num"],
                                    best_result["ner_recall"], best_result["correct_num"], best_result["golden_num"],
                                    best_result["type_f1"], best_result["type_precision"],
                                    best_result["type_correct_num"], best_result["type_infer_num"],
                                    best_result["type_recall"], best_result["type_correct_num"], best_result["type_golden_num"]))
                        else:
                            logger.info(
                                "best model: epoch {}, step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(
                                    best_result["epoch"], best_result["step"],
                                    best_result["ner_f1"], best_result["ner_precision"],
                                    best_result["correct_num"], best_result["infer_num"],
                                    best_result["ner_recall"], best_result["correct_num"], best_result["golden_num"]))
                    if args.eval_test:
                        eval_result = evaluate(
                            args, test_model, tokenizer, processor=processor, input_file=args.test_set, data_type='test', joiner_symbol="")
                        logger.info("[test], f1: {}\n".format(
                            eval_result['f1']))
                        writer.add_scalar(
                            "f1/test", eval_result["f1"], global_step)
                        if args.use_auxiliary_task:
                            writer.add_scalar(
                                "ner_f1/test", eval_result["ner_f1"], global_step)
                            writer.add_scalar(
                                "detection_f1/test", eval_result["type_f1"], global_step)

                    model.train()
                    # manually release the unused cache
                    if 'cuda' in str(args.device):
                        torch.cuda.empty_cache()
            print("one epoch end,  total {:5.2f}s".format(
                time.time() - one_epoch_start_time))
        logger.info("finish training")
        if args.use_auxiliary_task:
            logger.info("best model: step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}); detection f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                        format(best_result["step"], best_result["ner_f1"], best_result["ner_precision"],
                               best_result["correct_num"], best_result["infer_num"], best_result[
                            "ner_recall"], best_result["correct_num"], best_result["golden_num"],
                            best_result["type_f1"], best_result["type_precision"],
                            best_result["type_correct_num"], best_result["type_infer_num"],
                            best_result["type_recall"], best_result["type_correct_num"], best_result["type_golden_num"], time.time() - init_time))
        else:
            logger.info("best model: step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                        format(best_result["step"], best_result["ner_f1"], best_result["ner_precision"],
                               best_result["correct_num"], best_result["infer_num"], best_result[
                            "ner_recall"], best_result["correct_num"], best_result["golden_num"], time.time() - init_time))


def main():
    faulthandler.enable()
    args = get_argparse().parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and args.overwrite_output_dir:
        shutil.rmtree(os.path.join(args.output_dir, "tensorboard"))
        os.mkdir(os.path.join(args.output_dir, 'tensorboard'))
        # shutil.rmtree(os.path.join(args.output_dir, "logs"))
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
    if args.model_name == 'bert_ner_crf':
        model = BertNerCrf(args)
    elif args.model_name == 'bert_ner_softmax':
        model = BertNerSoftmax(args)
    model.to(args.device)

    # tokenizer = BertTokenizer.from_pretrained(
    #     args.model_name_or_path, do_lower_case=args.do_lower_case)

    tokenizer = BertWordPieceTokenizer(
        vocab=args.vocab_file, lowercase=args.do_lower_case)
    # tokenizer = BertWordPieceTokenizer(
    #     vocab="{}-vocab.txt".format(args.model_name_or_path), lowercase=args.do_lower_case)

    logger.info("Training/evaluation parameters %s", args)
    print_arguments(args, logger)
    if args.do_train:
        train(args, model, tokenizer, processor)
    if args.do_eval:
        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        load_model(model, args.output_dir, 'model.bin')

        # args.dev_set = '/home/yangpan/workspace/onepass_ie/data/ace05/dev'
        # args.test_set = '/home/yangpan/workspace/onepass_ie/data/ace05/test'

        if args.dev_set is not None:
            if os.path.isdir(args.dev_set):
                for dev_file in os.listdir(args.dev_set):
                    evaluate(args, model, tokenizer, processor,
                             output=False, output_eval_info=True, data_type='dev', input_file=os.path.join(args.dev_set, dev_file))
            else:
                evaluate(args, model, tokenizer, processor,
                         output=args.dump_result, output_eval_info=True, data_type='dev', input_file=args.dev_set)
        if args.test_set is not None:
            if os.path.isdir(args.test_set):
                for test_file in os.listdir(args.test_set):
                    evaluate(args, model, tokenizer, processor,
                             output=False, output_eval_info=True, data_type='test', input_file=os.path.join(args.test_set, test_file))
            else:
                evaluate(args, model, tokenizer, processor,
                         output=args.dump_result, output_eval_info=True, data_type='test', input_file=args.test_set)

# def trace(frame, event, arg):
#     print ("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#     return trace


if __name__ == "__main__":
    # sys.settrace(trace)
    main()
    # print("end")
