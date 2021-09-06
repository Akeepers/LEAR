import os
import shutil
from bertviz import head_view
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
# from models.model_ner import BertNerSpan, BertNerSpanMatrix, BertNerSpanMatrixWithLabelSiamese, BertNerSpanWithLabel, BertNerSpanWithLabelSiamese
import models.model_ner as models
import torch
import numpy as np
from utils.losses import *
from transformers import AdamW, BertTokenizerFast
from torch.optim import lr_scheduler
from callback.lr_scheduler import get_linear_schedule_with_warmup
from utils.common import EntityLabelWithScore, seed_everything, init_logger, logger, load_model, EntityLabel
import json
import time
from data_loader import NerDataProcessor
from utils.finetuning_args import get_argparse, print_arguments
from utils.evaluate import MetricsCalculator4Ner
import multiprocessing
import pickle
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


# g_best_params = None
# g_best_f1 = 0
# max_evals = 50

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


random_label_emb = None


def get_label_data(args, tokenizer):
    with open(args.label_str_file, 'r', encoding='utf-8') as fr:
        label_str_list = [line.strip() for line in fr.readlines()]
    token_ids, input_mask, max_len = [], [], 0
    for label_str in label_str_list:
        # sub_tokens = [tokenizer.cls_token] + \
        #     tokenizer.tokenize(label_str) + [tokenizer.sep_token]
        # token_id = tokenizer.convert_tokens_to_ids(sub_tokens)
        # input_mask.append([1] * len(token_id))
        encoded_results = tokenizer.encode_plus(label_str, add_special_tokens=True)
        token_id = encoded_results['input_ids']

        input_mask.append(encoded_results['attention_mask'])
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


def load_and_cache_examples(args, processor, input_file, data_type='train'):
    if os.path.exists(args.model_name_or_path):
        pretrain_model_name = str(args.model_name_or_path).split('/')[-1]
    else:
        pretrain_model_name = str(args.model_name_or_path)
    data_prefix = "".join(input_file.split("/")[-1].split(".")[:-1])
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(pretrain_model_name, args.data_name, data_prefix, str(
        args.task_type), ('uncased' if args.do_lower_case else 'cased'), str(args.max_seq_length if data_type == 'train' else args.max_seq_length)))
    if args.data_tag != "":
        cached_features_file += "_{}".format(args.data_tag)
    if args.sliding_len != -1:
        cached_features_file += "_slided{}".format(args.sliding_len)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'rb') as fr:
            results = pickle.load(fr)
            # results = {'features': features}
            logger.info("total records: {}, {}".format(len(results['features']), results['stat_info']))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # id2label, label2id = processor.get_labels(args.second_label_file)
        results = processor.convert_examples_to_feature(input_file, data_type)
        logger.info("Saving features into cached file {}, total_records: {}, {}".format(
                    cached_features_file, len(results['features']), results['stat_info']))
        # torch.save(features, cached_features_file)
        with open(cached_features_file, 'wb') as fw:
            pickle.dump(results, fw)
    return results['features']


def evaluate(args, model, processor, input_file, output=False, output_eval_info=False, data_type='dev'):
    dev_dataset = load_and_cache_examples(args, processor, input_file, data_type=data_type)
    dev_dataloader = DataLoaderX(dataset=dev_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=multiprocessing.cpu_count()//4,
                                 collate_fn=processor.generate_batch_data())

    metrics = MetricsCalculator4Ner(args, processor)
    if args.model_name == 'SERS':
        # label_token_ids, label_input_mask, label_token_type_ids = get_label_data(args, processor.get_tokenizer())
        # batch_label_token_ids = torch.tensor(label_token_ids, dtype=torch.long).to(args.device)
        # batch_label_input_mask = torch.tensor(label_input_mask, dtype=torch.float32).to(args.device)
        # batch_label_token_type_ids = torch.tensor(label_token_type_ids, dtype=torch.long).to(args.device)
        batch_label_token_ids, batch_label_token_type_ids, batch_label_input_mask = processor.get_label_data(args.device)

        global random_label_emb
        if args.use_random_label_emb and random_label_emb is None:
            random_label_emb = torch.rand(size=batch_label_token_ids.shape + (1024,))
    model.eval()

    dev_bar = tqdm(dev_dataloader, desc="Evaluation")
    for data in dev_bar:
        # for step, data in enumerate(dev_dataloader):
        with torch.no_grad():
            if args.use_random_label_emb:
                data['random_label_emb'] = random_label_emb
            for key in data.keys():
                if key not in ['golden_label', 'ids', 'seq_len']:
                    data[key] = data[key].to(args.device)

            if args.model_name == 'SERS':
                results = model(data, batch_label_token_ids,
                                batch_label_token_type_ids, batch_label_input_mask, return_score=args.dump_result, mode='inference', return_bert_attention=args.visualizate_bert)
                infer_starts, infer_ends = torch.sigmoid(results[0]), torch.sigmoid(results[1])
                if args.visualizate_bert:
                    # print(type(processor.get_tokenizer().decode(data['token_ids'][0])))
                    # print(results[-1])
                    head_view(results[-1], processor.get_tokenizer().decode(data['token_ids'][0]).split(' '))
                if args.exist_nested:
                    infer_mathches = torch.sigmoid(results[2])
            else:
                results = model(data)
                infer_starts, infer_ends = torch.sigmoid(results[0]), torch.sigmoid(results[1])
                if args.exist_nested:
                    infer_mathches = torch.sigmoid(results[2])
            # metrics.update(results[0].cpu().numpy(), results[1].cpu().numpy(), data['golden_label'], data['seq_len'],
            #                match_label_ids=(results[2] if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))
            # metrics.update(results[0], results[1], data['golden_label'], data['seq_len'],
            #                match_label_ids=(results[2] if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))
            # metrics.update(infer_starts.cpu().numpy(), infer_ends.cpu().numpy(), data['golden_label'], data['seq_len'].cpu().numpy(),
            #                match_label_ids=(infer_mathches.cpu().numpy() if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))
            metrics.update(infer_starts.cpu(), infer_ends.cpu(), data['golden_label'], data['seq_len'].cpu(),
                           match_label_ids=(infer_mathches.cpu() if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))

    if output:
        # check the result dir
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        result_list = metrics.get_results()
        path = os.path.join(args.result_dir, "{}_result_ner.json".format(data_type))
        with open(path, 'w', encoding='utf-8') as fw:
            for line in result_list:
                fw.write(json.dumps(line, indent=4, ensure_ascii=False) + '\n')

    result_dict = metrics.get_metrics()['general']
    if output_eval_info:
        data_prefix = input_file.split('/')[-1].split('.')[0]
        logger.info("***** Eval results: {} *****".format(data_prefix))
        logger.info(
            "f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(result_dict["f1"], result_dict["precision"],
                                                                       result_dict["correct_num"], result_dict["infer_num"], result_dict[
                "recall"], result_dict["correct_num"], result_dict["golden_num"]))

    return result_dict


def analysis(args, model, processor, input_file, output=False, output_eval_info=False, data_type='dev'):
    dev_dataset = load_and_cache_examples(args, processor, input_file, data_type=data_type)
    dev_dataloader = DataLoaderX(dataset=dev_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=multiprocessing.cpu_count()//4,
                                 collate_fn=processor.generate_batch_data())

    metrics = MetricsCalculator4Ner(args, processor)
    if args.model_name == 'SERS':
        batch_label_token_ids, batch_label_token_type_ids, batch_label_input_mask = processor.get_label_data(args.device)

        global random_label_emb
        if args.use_random_label_emb and random_label_emb is None:
            random_label_emb = torch.rand(size=batch_label_token_ids.shape + (1024,))
    model.eval()

    dev_bar = tqdm(dev_dataloader, desc="Evaluation")
    for data in dev_bar:
        # for step, data in enumerate(dev_dataloader):
        with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1), profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.output_dir, "tensorboard"))) as profiler_utlility:

            with torch.no_grad():
                if args.use_random_label_emb:
                    data['random_label_emb'] = random_label_emb
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'seq_len']:
                        data[key] = data[key].to(args.device)

                if args.model_name == 'SERS':
                    results = model(data, batch_label_token_ids,
                                    batch_label_token_type_ids, batch_label_input_mask, return_score=args.dump_result, mode='inference', return_bert_attention=args.visualizate_bert)
                    infer_starts, infer_ends = torch.sigmoid(results[0]), torch.sigmoid(results[1])
                    if args.visualizate_bert:
                        # print(type(processor.get_tokenizer().decode(data['token_ids'][0])))
                        # print(results[-1])
                        head_view(results[-1], processor.get_tokenizer().decode(data['token_ids'][0]).split(' '))
                    if args.exist_nested:
                        infer_mathches = torch.sigmoid(results[2])
                else:
                    results = model(data)
                    infer_starts, infer_ends = torch.sigmoid(results[0]), torch.sigmoid(results[1])
                    if args.exist_nested:
                        infer_mathches = torch.sigmoid(results[2])
                profiler_utlility.step()
                # metrics.update(results[0].cpu().numpy(), results[1].cpu().numpy(), data['golden_label'], data['seq_len'],
                #                match_label_ids=(results[2] if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))
                # metrics.update(results[0], results[1], data['golden_label'], data['seq_len'],
                #                match_label_ids=(results[2] if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))
                # metrics.update(infer_starts.cpu().numpy(), infer_ends.cpu().numpy(), data['golden_label'], data['seq_len'].cpu().numpy(),
                #                match_label_ids=(infer_mathches.cpu().numpy() if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))
                metrics.update(infer_starts.cpu(), infer_ends.cpu(), data['golden_label'], data['seq_len'].cpu(),
                               match_label_ids=(infer_mathches.cpu() if args.exist_nested else None), is_logits=False, tokens=(data['token_ids'] if output else None))

    if output:
        # check the result dir
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        result_list = metrics.get_results()
        path = os.path.join(args.result_dir, "{}_result_ner.json".format(data_type))
        with open(path, 'w', encoding='utf-8') as fw:
            for line in result_list:
                fw.write(json.dumps(line, indent=4, ensure_ascii=False) + '\n')

    result_dict = metrics.get_metrics()['general']
    if output_eval_info:
        data_prefix = input_file.split('/')[-1].split('.')[0]
        logger.info("***** Eval results: {} *****".format(data_prefix))
        logger.info(
            "f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(result_dict["f1"], result_dict["precision"],
                                                                       result_dict["correct_num"], result_dict["infer_num"], result_dict[
                "recall"], result_dict["correct_num"], result_dict["golden_num"]))

    return result_dict


def train(args, model, tokenizer, processor):

    if args.do_ema:
        ema = ModelEma(model, 0.9997)

    # weight_sum = args.weight_start_loss + \
    #     args.weight_end_loss + args.weight_span_loss
    # args.weight_start_loss = args.weight_start_loss / weight_sum
    # args.weight_end_loss = args.weight_end_loss / weight_sum
    # args.weight_span_loss = args.weight_span_loss / weight_sum

    # print("{}, {}, {}".format(args.weight_start_loss, args.weight_end_loss, args.weight_span_loss))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, data_type="train", processor=processor, input_file=args.train_set)
    train_dataloader = DataLoaderX(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=args.drop_last,
                                   #    num_workers=multiprocessing.cpu_count()//4,
                                   num_workers=4,
                                   collate_fn=processor.generate_batch_data())

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.val_step = max(1, len(train_dataloader) // args.eval_per_epoch)
    if args.val_skip_epoch > 0:
        args.val_skip_step = max(
            1, len(train_dataloader)) * args.val_skip_epoch
    # define the optimizer

    bert_parameters = model.bert.named_parameters()
    first_start_params = model.entity_start_classifier.named_parameters()
    first_end_params = model.entity_end_classifier.named_parameters()

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

    if args.exist_nested:
        span_matir_params = model.span_embedding.named_parameters()
        optimizer_grouped_parameters += [
            {"params": [p for n, p in span_matir_params if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
            {"params": [p for n, p in span_matir_params if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
        ]
    if args.model_name == 'SERS':
        label_fused_params = model.label_fusing_layer.named_parameters()
        optimizer_grouped_parameters += [
            {"params": [p for n, p in label_fused_params if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.learning_rate * args.task_layer_lr},
            {"params": [p for n, p in label_fused_params if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate * args.task_layer_lr},
        ]

    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    if args.do_ema:
        steps_per_epoch = len(train_dataloader)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=steps_per_epoch,
                                            epochs=int(args.num_train_epochs), pct_start=0.2)
    else:
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
    # todo 测试python3，这里的seed是否仍然需要
    seed_everything(args.seed)

    if args.model_name == 'SERS':
        # label_token_ids, label_input_mask, label_token_type_ids = get_label_data(
        #     args, tokenizer)

        # batch_label_token_ids = torch.tensor(
        #     label_token_ids, dtype=torch.long).to(args.device)
        # batch_label_input_mask = torch.tensor(
        #     label_input_mask, dtype=torch.float32).to(args.device)
        # batch_label_token_type_ids = torch.tensor(
        #     label_token_type_ids, dtype=torch.long).to(args.device)
        # print(batch_label_token_ids.shape)
        batch_label_token_ids, batch_label_token_type_ids, batch_label_input_mask = processor.get_label_data(args.device)
        global random_label_emb
        if args.use_random_label_emb and random_label_emb is None:
            random_label_emb = torch.rand(size=batch_label_token_ids.shape + (1024,))

    if args.do_train:
        model.train()
        global_step = 0

        best_result = {'f1': 0.0}
        init_time = time.time()

        # the training loop
        add_label_info = True
        # profile_path = os.path.join(args.output_dir, "tensorboard")
        # print(profile_path)
        # with torch.profiler.profile(activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1), profile_memory=False, on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path)) as profiler_utlility:
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA],
        #     profile_memory=True
        # ) as p:
        for epoch in range(int(args.num_train_epochs)):
            train_bar = tqdm(train_dataloader, desc="Training")
            for cur_step, data in enumerate(train_bar):
                global_step += 1

                if args.use_random_label_emb:
                    data['random_label_emb'] = random_label_emb
                for key in data.keys():
                    if key not in ['golden_label', 'ids', 'seq_len']:
                        data[key] = data[key].to(args.device)
                if args.model_name == 'SERS':
                    results = model(data, batch_label_token_ids, batch_label_token_type_ids,
                                    batch_label_input_mask,
                                    # add_label_info=add_label_info
                                    )
                else:
                    results = model(data)

                if args.exist_nested:
                    start_logits, end_logits, match_logits = results
                    # start_loss, end_loss, span_loss = compute_focal_loss_v2(
                    #     data['first_starts'], start_logits, data['first_ends'], end_logits, data['match_label'], match_logits, data['input_mask'])
                    if args.use_focal_loss:
                        start_loss, end_loss, span_loss = compute_focal_loss_v2(
                            data['first_starts'], start_logits, data['first_ends'], end_logits, data['match_label'], match_logits, data['input_mask'], alpha=args.alpha, gamma=args.gamma)
                    else:
                        start_loss, end_loss, span_loss = compute_loss_v2(
                            data['first_starts'], start_logits, data['first_ends'], end_logits, data['match_label'], match_logits, data['input_mask'])
                    span_loss_ = span_loss * args.weight_span_loss
                    start_loss_ = start_loss * args.weight_start_loss
                    end_loss_ = end_loss * args.weight_end_loss
                    total_loss = span_loss_ + start_loss_ + end_loss_

                    writer.add_scalars('loss/train', {'total_loss': total_loss.item(), 'span_loss': span_loss_.item(),
                                                      'start_loss': start_loss_.item(), 'end_loss': end_loss_.item()}, global_step)
                else:
                    start_logits, end_logits = results[:2]
                    start_loss = loss_v2(data['first_starts'], start_logits, data['input_mask'])
                    end_loss = loss_v2(data['first_ends'], end_logits, data['input_mask'])
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
                train_bar.set_description(
                    "{}/{} step:{}, loss:{:.6}".format(epoch + 1, int(args.num_train_epochs), global_step, total_loss.item()))

                # add_label_info = False

                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    if args.do_ema:
                        ema.update(model)
                    # add_label_info = True
                    # print(global_step)

                if args.do_eval and global_step > args.val_skip_step and global_step % args.val_step == 0:
                    model.eval()
                    test_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    # call the test function

                    eval_result = evaluate(args, test_model, processor=processor, input_file=args.dev_set, data_type='dev', output_eval_info=True)
                    writer.add_scalar(
                        "f1/dev", eval_result["f1"], global_step)
                    logger.info("[dev], f1: {}\n".format(eval_result['f1']))
                    if eval_result["f1"] > best_result["f1"]:
                        best_result.update(eval_result)
                        best_result["step"] = global_step
                        best_result['epoch'] = epoch + 1
                        # save the best model
                        output_dir = args.output_dir
                        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

                    if eval_result["f1"] > 0:
                        logger.info(
                            "best model: epoch {}, step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{})".format(
                                best_result["epoch"], best_result["step"],
                                best_result["f1"], best_result["precision"],
                                best_result["correct_num"], best_result["infer_num"],
                                best_result["recall"], best_result["correct_num"], best_result["golden_num"]))
                    if args.do_ema:
                        ema_results = evaluate(args, ema.module, processor=processor, input_file=args.dev_set,
                                               data_type='dev', output_eval_info=False)
                        logger.info("ema result [dev]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                                    format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))
                        writer.add_scalar("ema_f1/dev", ema_results["f1"], global_step)
                    if args.eval_test:
                        eval_result = evaluate(args, test_model, processor=processor, input_file=args.test_set, data_type='test')
                        logger.info("[test], f1: {}\n".format(eval_result['f1']))
                        writer.add_scalar("f1/test", eval_result["f1"], global_step)

                        if args.do_ema:
                            ema_results = evaluate(args, ema.module, processor=processor, input_file=args.test_set,
                                                   data_type='test', output_eval_info=False)
                            logger.info("ema result [test]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                                        format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))
                            writer.add_scalar("ema_f1/test", ema_results["f1"], global_step)
                    model.train()

                # profiler_utlility.step()
                # manually release the unused cache
                # if 'cuda' in str(args.device):
                #     torch.cuda.empty_cache()
            # print("one epoch end,  total {:5.2f}s".format(
            #     time.time() - one_epoch_start_time))
        # print(p.key_averages().table(
        #     sort_by="self_cuda_time_total", row_limit=-1))
        logger.info("** finish training **\n")
        logger.info("best model: step {}, -- f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                    format(best_result["step"], best_result["f1"], best_result["precision"], best_result["correct_num"], best_result["infer_num"], best_result["recall"], best_result["correct_num"], best_result["golden_num"], time.time() - init_time))
        if args.do_ema:
            ema_results = evaluate(args, ema.module, processor=processor, input_file=args.dev_set, data_type='dev', output_eval_info=True)
            logger.info("ema result [dev]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                        format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))
            if ema_results["f1"] > best_result["f1"]:
                # save the best model
                output_dir = args.output_dir
                model_to_save = ema.module
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

            ema_results = evaluate(args, ema.module, processor=processor, input_file=args.test_set, data_type='dev', output_eval_info=True)
            logger.info("ema result [test]: f1: {:4.4f}, p: {:4.4f}({}/{}), r: {:4.4f}({}/{}), total time: {:5.2f}s".
                        format(ema_results["f1"], ema_results["precision"], ema_results["correct_num"], ema_results["infer_num"], ema_results["recall"], ema_results["correct_num"], ema_results["golden_num"], time.time() - init_time))


def main(args):
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

    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, 'logs'))
        os.mkdir(os.path.join(args.output_dir, 'tensorboard'))
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # init_logger(log_file=args.output_dir + f'/logs/{args.task_type}-{time_}.log')
    init_logger(log_file="{}/logs/{}_{}-{}.log".format(args.output_dir, ('train' if args.do_train else 'eval'), args.task_type, time_))

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    processor = NerDataProcessor(args, tokenizer)
    args.first_label_num = processor.get_class_num()
    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)
    print_arguments(args, logger)

    # init_model
    if args.model_name == 'bert_ner':
        model = models.BertNerSpanMatrix(args) if args.exist_nested else models.BertNerSpan(args)
    elif args.model_name == 'SERS':
        model = models.LEARNer4Nested(args) if args.exist_nested else models.LEARNer4Flat(args)
    else:
        pass
    model.to(args.device)

    if args.do_train:
        train(args, model, tokenizer, processor)
    if args.do_eval:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        load_model(model, args.output_dir, 'model.bin')

        # args.dev_set = '/home/yangpan/workspace/onepass_ie/data/ace05/dev'
        # args.test_set = '/home/yangpan/workspace/onepass_ie/data/ace05/test'

        if args.dev_set is not None:
            if os.path.isdir(args.dev_set):
                for dev_file in os.listdir(args.dev_set):
                    evaluate(args, model, processor, output=args.dump_result, output_eval_info=True,
                             data_type='dev', input_file=os.path.join(args.dev_set, dev_file))
            else:
                evaluate(args, model, processor, output=args.dump_result, output_eval_info=True, data_type='dev', input_file=args.dev_set)
        if args.test_set is not None:
            if os.path.isdir(args.test_set):
                for test_file in os.listdir(args.test_set):
                    evaluate(args, model, processor,
                             output=args.dump_result, output_eval_info=True, data_type='test', input_file=os.path.join(args.test_set, test_file))
            else:
                results = evaluate(args, model, processor, output=args.dump_result, output_eval_info=True, data_type='test', input_file=args.test_set)
                # global g_best_f1
                # if results['f1'] > g_best_f1:
                #     g_best_f1 = results['f1']
                #     g_best_params = args
                # print('*** cur best: {}'.format(g_best_f1))
                # print_arguments(args, logger)
                # print('\n')
# def trace(frame, event, arg):
#     print ("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#     return trace


if __name__ == "__main__":
    # sys.settrace(trace)
    args = get_argparse().parse_args()
    main(args)

    # param_grid = {
    #     'num_train_epochs': list(range(5, 25, 2)),
    #     # 'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    #     'learning_rate': list(np.arange(1e-6, 4e-5, 1e-6)),
    #     'batch_size': [8, 16, 24, 32],
    #     'task_layer_lr': list(range(5, 51, 5))
    # }

    # for i in range(max_evals):
    #     random.seed(i)
    #     hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    #     args.seed = i
    #     args.num_train_epochs = hyperparameters['num_train_epochs']
    #     args.learning_rate = hyperparameters['learning_rate']
    #     args.task_layer_lr = hyperparameters['task_layer_lr']
    #     args.per_gpu_train_batch_size = hyperparameters['batch_size']
    #     main(args)
    # print("end")
