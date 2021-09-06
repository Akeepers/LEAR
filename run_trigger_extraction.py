
import os
import argparse
import pickle
from tqdm import tqdm
from numpy.lib.function_base import extract
import models.model_event as models
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import BertWordPieceTokenizer
import torch
import numpy as np
from transformers import AdamW, BertTokenizerFast, RobertaTokenizerFast
from callback.lr_scheduler import get_linear_schedule_with_warmup
from utils.common import seed_everything, init_logger, logger, load_model, EventTypeLabel, EntityLabelWithScore
from utils.losses import *
import json
import jsonlines
import time
from data_loader import TriggerDataProcessor
from utils.finetuning_args import get_argparse, print_arguments
from utils.evaluate import MetricsCalculator4ED
from prefetch_generator import BackgroundGenerator
import multiprocessing


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_and_cache_examples(args, processor, input_file, data_type='train'):
    pretrain_model_name = str(args.model_name_or_path).split('/')[-1] if os.path.exists(args.model_name_or_path) else str(args.model_name_or_path)
    if '/' in pretrain_model_name:
        pretrain_model_name = pretrain_model_name.split('/')[-1]
    data_prefix = "".join(input_file.split("/")[-1].split(".")[:-1])

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
        pretrain_model_name,
        args.data_name,
        data_prefix,
        str(args.task_type),
        ('uncased' if args.do_lower_case else 'cased'),
        str(args.max_seq_length)))
    if args.data_tag != "":
        cached_features_file += "_{}".format(args.data_tag)

    # load cached dataset
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file '{}'".format(cached_features_file))
        with open(cached_features_file, 'rb') as fr:
            results = pickle.load(fr)
            logger.info("total records: {}, {}".format(len(results['features']), results['stat_info']))
    else:
        # load & save dataset
        logger.info("Creating features from dataset file at %s", args.data_dir)
        results = processor.convert_examples_to_feature(input_file, data_type)
        logger.info("Saving features into cached file {}, total_records: {}, {}".format(
            cached_features_file, len(results['features']), results['stat_info']))
        with open(cached_features_file, 'wb') as fw:
            pickle.dump(results, fw)
    return results['features']


def evaluate(args, model, processor, input_file, cal_metrics=True, output=False, output_eval_info=False, data_type='dev'):
    dataset = load_and_cache_examples(args, processor, data_type=data_type, input_file=input_file)
    eval_dataloader = DataLoader(dataset=dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=4,
                                 collate_fn=processor.generate_batch_data())
    metrics = MetricsCalculator4ED(args, processor)
    model.eval()

    if args.test_speed:
        eval_start_time = time.time()
    with tqdm(eval_dataloader, desc="Evaluation") as eval_bar:
        if args.model_name == "LEAR":
            with torch.no_grad():
                label_token_ids, label_token_type_ids, label_input_mask = processor.get_label_data(args.device)
                label_embs = model.get_text_embedding(label_token_ids, label_token_type_ids, label_input_mask,
                                                      return_token_level=True)['token_level_embs']
        for step, data in enumerate(eval_bar):
            with torch.no_grad():
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'seq_len']:
                        data[key] = data[key].to(args.device)
                if args.model_name == "LEAR":
                    data['label_embs'] = label_embs
                    data['label_input_mask'] = label_input_mask
                    results = model(data, return_score=args.dump_result, mode='inference')
                else:
                    results = model(data)
                if not args.test_speed:
                    infer_trigger_starts, infer_trigger_ends = results[:2]
                    metrics.update(infer_trigger_starts.cpu(), infer_trigger_ends.cpu(), data['golden_label'], data['seq_len'], is_logits=False,
                                   match_pattern=args.match_pattern,
                                   tokens=(data['token_ids'] if output else None),
                                   ids=(data['ids'] if args.data_type == 'maven' else None))

    if args.test_speed:
        eval_total_time = time.time() - eval_start_time
    if output:
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        result_list = metrics.get_results(append_ids=(args.data_type == 'maven'))
        if args.data_type == 'maven':
            path = os.path.join(args.result_dir, "{}_result_trigger.txt".format(data_type))
            with open(path, 'w', encoding='utf-8') as fw:
                for line in result_list:
                    fw.write(json.dumps(line) + '\n')
        else:
            path = os.path.join(args.result_dir, "{}_result_trigger.json".format(data_type))
            with open(path, 'w', encoding='utf-8') as fw:
                for line in result_list:
                    fw.write(json.dumps(line, indent=4, ensure_ascii=False) + '\n')
    result_dict = {}
    if cal_metrics:
        result_dict.update(metrics.get_metrics()['general'])
    if args.test_speed:
        result_dict['eval_time'] = eval_total_time
    if output_eval_info:
        data_prefix = input_file.split('/')[-1].split('.')[0]
        logger.info("***** Eval results: {} *****".format(data_prefix))
        logger.info("trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".
                    format(result_dict["f1"], result_dict["precision"], result_dict["correct_num"], result_dict["infer_num"], result_dict["recall"], result_dict["correct_num"], result_dict["golden_num"]))
    return result_dict


def train(args, model, processor):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # args.eval_test = False if args.is_maven else args.eval_test
    train_dataset = load_and_cache_examples(args, processor=processor, data_type="train", input_file=args.train_set)
    train_dataloader = DataLoaderX(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True,
                                   num_workers=4,
                                   collate_fn=processor.generate_batch_data())

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.val_step = max(1, len(train_dataloader) // args.eval_per_epoch)
    if args.val_skip_epoch > 0:
        args.val_skip_step = max(1, len(train_dataloader)) * args.val_skip_epoch

    # define the optimizer
    bert_parameters = model.bert.named_parameters()
    first_start_params = model.first_head_classifier.named_parameters()
    first_end_params = model.first_tail_classifier.named_parameters()

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate},
        {"params": [p for n, p in first_start_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in first_start_params if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in first_end_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
        {"params": [p for n, p in first_end_params if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
    ]
    if args.model_name == "LEAR":
        label_fused_params = model.label_fusing_layer.named_parameters()
        optimizer_grouped_parameters += [
            {"params": [p for n, p in label_fused_params if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
            {"params": [p for n, p in label_fused_params if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr}, ]

    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

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

    if args.do_train:
        model.train()
        global_step = 0

        best_result = {'f1': 0.0}

        init_time = time.time()

        # the training loop
        add_label_info = True
        if args.test_speed:
            train_total_time, train_cnt = 0.0, 0
            eval_total_time, eval_cnt = 0.0, 0
        for epoch in range(int(args.num_train_epochs)):
            train_bar = tqdm(train_dataloader, desc="Training")
            one_epoch_start_time = time.time()
            for data in train_bar:
                global_step += 1
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'seq_len']:
                        data[key] = data[key].to(args.device)
                if args.model_name == "LEAR":
                    data['label_token_ids'], data['label_token_type_ids'], data['label_input_mask'] = processor.get_label_data(args.device)
                    pred_sub_heads, pred_sub_tails = model(data, add_label_info=add_label_info)
                else:
                    pred_sub_heads, pred_sub_tails = model(data)

                start_loss = loss_v2(data['first_starts'], pred_sub_heads, data['input_mask'], is_logit=False)
                end_loss = loss_v2(data['first_ends'], pred_sub_tails, data['input_mask'], is_logit=False)
                total_loss = (start_loss + end_loss)

                writer.add_scalars(
                    'loss/train', {
                        'total_loss': total_loss.item(),
                        'start_loss': start_loss.item(),
                        'end_loss': end_loss.item()
                    },
                    global_step,
                )

                if args.n_gpu > 1:
                    total_loss = total_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

                total_loss.backward()
                train_bar.set_description("epoch:{}/{} - step:{}, loss:{:.6f} ".format(epoch + 1, int(args.num_train_epochs),
                                          global_step, total_loss.item() * args.gradient_accumulation_steps))

                add_label_info = False
                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    add_label_info = True

                if args.do_eval and global_step > args.val_skip_step and global_step % args.val_step == 0:
                    if args.test_speed:
                        eval_start_time = time.time()
                    test_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    # call the test function

                    eval_result = evaluate(args, test_model, processor=processor, input_file=args.dev_set, data_type='dev', output_eval_info=False)

                    writer.add_scalar("trigger_f1/dev", eval_result["f1"], global_step)
                    writer.add_scalar("trigger_precision/dev", eval_result["precision"], global_step)
                    if args.test_speed:
                        eval_cnt += 1
                        eval_total_time += eval_result['eval_time']
                        logger.info("one eval end, cost {:5.2f}s, average {:5.2f}, cnt: {}".format(
                            eval_result['eval_time'], eval_total_time/eval_cnt, eval_cnt))

                    if eval_result["f1"] > best_result["f1"]:
                        best_result.update(eval_result)
                        best_result["step"] = global_step
                        best_result["epoch"] = epoch + 1
                        # save the best model
                        output_dir = args.output_dir
                        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    logger.info("**Dev**")
                    logger.info("f1 - {}".format(eval_result['f1']))
                    if eval_result["f1"] > 0:
                        logger.info(
                            "best model: step {}, epoch: {} , trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{});".
                            format(best_result["step"], best_result['epoch'], best_result["f1"], best_result["precision"],
                                   best_result["correct_num"], best_result["infer_num"], best_result["recall"], best_result["correct_num"], best_result["golden_num"]))
                    if args.eval_test:
                        eval_result = evaluate(args, test_model, processor=processor, input_file=args.test_set, data_type='test')
                        writer.add_scalar("trigger_f1/test", eval_result["trigger_f1"], global_step)
                        writer.add_scalar("trigger_precision/test", eval_result["precision"], global_step)
                        writer.add_scalar("trigger_recall/test", eval_result["recall"], global_step)
                        logger.info("**Test**")
                        logger.info("f1 - {}".format(eval_result))
                    model.train()

                    # manually release the unused cache
                    # if 'cuda' in str(args.device):
                    #     torch.cuda.empty_cache()

            if args.test_speed:
                one_epoch_train_time = time.time() - one_epoch_start_time

                train_cnt += 1
                train_total_time += one_epoch_train_time
                logger.info("one epoch end, cost {:5.2f}s, average {:5.2f}, cnt: {}".format(
                    one_epoch_train_time, train_total_time/train_cnt, train_cnt))

        logger.info("finish training")
        logger.info(
            "best model: step {}, epoch: {} , trigger f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{});".
            format(best_result["step"], best_result['epoch'], best_result["f1"], best_result["precision"],
                   best_result["correct_num"], best_result["infer_num"], best_result["recall"], best_result["correct_num"], best_result["golden_num"]))


def main():
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if args.model_name_or_path.endswith('-uncased') and (not args.do_lower_case):
        raise ValueError("use uncased model, 'do_lower_case' must be True")
    if args.model_name_or_path.endswith('-cased') and args.do_lower_case:
        raise ValueError("use cased model, 'do_lower_case' must be False")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, 'logs'))
        os.mkdir(os.path.join(args.output_dir, 'tensorboard'))
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/logs/{args.task_type}-{time_}.log')
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.pretrain_model_name = args.model_name_or_path.split('-')[0].split('/')[-1]
    if "bert" == args.pretrain_model_name or args.pretrain_model_name == 'spanbert':
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    elif "roberta" == args.pretrain_model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name_or_path)
    processor = TriggerDataProcessor(args, tokenizer)
    args.first_label_num = processor.get_class_num()

    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)
    print_arguments(args, logger)

    # init_model
    if args.model_name == 'bert_span_ed':
        model = models.BertSpan4ED(args)
    elif args.model_name == "LEAR":
        model = models.LEAR4ED(args)
    model.to(args.device)

    if args.do_train:
        train(args, model, processor)
    if args.do_eval:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        load_model(model, args.output_dir, 'model.bin')

        if args.data_type == 'maven':
            evaluate(args, model, processor, cal_metrics=False, output=args.dump_result,
                     output_eval_info=False, data_type='test', input_file=args.test_set)
        else:
            if args.dev_set is not None:
                if os.path.isdir(args.dev_set):
                    for dev_file in os.listdir(args.dev_set):
                        evaluate(args, model, processor, output=False, output_eval_info=True,
                                 data_type='dev', input_file=os.path.join(args.dev_set, dev_file))
                else:
                    evaluate(args, model, processor, output=args.dump_result, output_eval_info=True, data_type='dev', input_file=args.dev_set)
            if args.test_set is not None:
                if os.path.isdir(args.test_set):
                    for test_file in os.listdir(args.test_set):
                        evaluate(args, model, processor, output=False, output_eval_info=True,
                                 data_type='test', input_file=os.path.join(args.test_set, test_file))
                else:
                    evaluate(args, model, processor, output=args.dump_result, output_eval_info=True, data_type='test', input_file=args.test_set)


if __name__ == "__main__":
    args = get_argparse().parse_args()
    main()
