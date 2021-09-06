import argparse
import six


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, positional_arg=False, **kwargs):
        prefix = "" if positional_arg else "--"
        type = str2bool if type == bool else type
        self._group.add_argument(
            prefix + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args, log):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')


def get_argparse():
    parser = argparse.ArgumentParser()
    model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
    # Required parameters
    model_g.add_arg("task_type", default="sequence_classification", type=str, required=True,
                    help="The type of the task to train selected in the list: ")
    model_g.add_arg("span_decode_strategy", default="v3", type=str, required=False, help="")
    model_g.add_arg("task_save_name", default="", type=str, required=True, help="")
    model_g.add_arg("data_dir", default=None, type=str, required=True, help="The input data dir. ")
    model_g.add_arg("data_name", default=None, type=str, required=True, help="")
    model_g.add_arg("result_dir", default=None, type=str, required=False, help="")
    model_g.add_arg("model_name", default="casrel", type=str, required=True, help="Model type selected in the list: ")
    model_g.add_arg("model_name_or_path", default='bert-base-cased', type=str, required=True,
                    help="Path to pre-trained model or shortcut name selected in the list: ")
    model_g.add_arg("output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
    model_g.add_arg("checkpoint", default=None, type=str, required=False, help="")
    model_g.add_arg("train_set", default=None, type=str, required=False, help="")
    model_g.add_arg("dev_set", default=None, type=str, required=False, help="")
    model_g.add_arg("vocab_file", default=None, type=str, required=False, help="")
    model_g.add_arg("test_set", default=None, type=str, required=False, help="")
    model_g.add_arg("first_label_file", default=None, type=str, required=False, help="")
    model_g.add_arg("second_label_file", default=None, type=str, required=False, help="")
    model_g.add_arg("label_str_file", default=None, type=str, required=False, help="")
    model_g.add_arg("glove_label_emb_file", default=None, type=str, required=False, help="")
    model_g.add_arg("label_ann_vocab_file", default=None, type=str, required=False, help="")
    model_g.add_arg("label_ann_word_id_list_file", default=None, type=str, required=False, help="")
    model_g.add_arg("data_tag", default="", type=str, required=False, help="")
    model_g.add_arg("use_auxiliary_task", default=False, type=bool, help="Whether to transform first_label feature.")
    model_g.add_arg("exist_nested", default=False, type=bool, help="")
    model_g.add_arg("use_attn", default=False, type=bool, help="")
    model_g.add_arg("gradient_checkpointing", default=False, type=bool, help="")
    model_g.add_arg("use_random_label_emb", default=False, type=bool, help="")
    model_g.add_arg("use_label_encoding", default=False, type=bool, help="")
    model_g.add_arg("label_list", default="", type=str, required=False, help="")
    model_g.add_arg("dump_result", default=False, type=bool, help="")
    model_g.add_arg("sliding_len", default=-1, type=int, help="")
    model_g.add_arg("weight_start_loss", default=1.0, type=float, help="")
    model_g.add_arg("weight_end_loss", default=1.0, type=float, help="")
    model_g.add_arg("weight_span_loss", default=1.0, type=float, help="")
    # Other parameters
    model_g.add_arg('loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'], help="")
    model_g.add_arg("max_seq_length", default=256, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.", )
    model_g.add_arg("do_train", type=bool, default=True, help="Whether to run training.")
    model_g.add_arg("eval_test", type=bool, default=False, help="Whether to run training.")
    model_g.add_arg("do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    model_g.add_arg("visualizate_bert", type=bool, default=False, help="")
    model_g.add_arg("drop_last", type=bool, default=True, help="")
    model_g.add_arg("padding_to_max", type=bool, default=False, help="")
    model_g.add_arg("is_chinese", type=bool, default=False, help="")
    model_g.add_arg("do_ema", type=bool, default=False, help="")
    model_g.add_arg("data_type", type=str, default="default", help="")
    model_g.add_arg("match_pattern", type=str, default="default", help="")
    model_g.add_arg("test_speed", type=bool, default=False, help="")
    model_g.add_arg("use_focal_loss", type=bool, default=False, help="")
    model_g.add_arg("alpha", default=0.25, type=float, help="")
    model_g.add_arg("gamma", default=2, type=int, help="")
    model_g.add_arg("label_ann_vocab_size", default=-1, type=int, help="")
    model_g.add_arg("do_add", type=bool, default=False, help="")
    model_g.add_arg("average_pooling", type=bool, default=False, help="")
    model_g.add_arg("use_label_embedding", type=bool, default=False, help="")
    model_g.add_arg("do_predict", type=bool, default=True, help="Whether to run predictions on the test set.")
    model_g.add_arg("evaluate_during_training", type=bool, default=True, help="Whether to run evaluation during training at each logging step.", )
    model_g.add_arg("do_lower_case", type=bool, default=True, help="Set this flag if you are using an uncased model.")
    model_g.add_arg("per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    model_g.add_arg("val_step", default=100, type=int, help="")
    model_g.add_arg("val_skip_step", default=1, type=int, help="")
    model_g.add_arg("val_skip_epoch", default=0, type=int, help="")
    model_g.add_arg("label_emb_size", default=300, type=int, help="")
    model_g.add_arg("first_label_num", default=1, type=int, help="")
    model_g.add_arg("second_label_num", default=1, type=int, help="")
    model_g.add_arg("eval_per_epoch", default=1, type=int, help="")
    model_g.add_arg("per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    model_g.add_arg("gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.", )
    model_g.add_arg("learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    model_g.add_arg("task_layer_lr", default=20, type=float, help=".")
    model_g.add_arg("weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    model_g.add_arg("adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    model_g.add_arg("dropout_rate", default=0.1, type=float, help="")
    model_g.add_arg("classifier_dropout_rate", default=0.1, type=float, help="")
    model_g.add_arg("max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    model_g.add_arg("num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    model_g.add_arg("max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    model_g.add_arg("warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    model_g.add_arg("logging_steps", type=int, default=50, help="Log every X updates steps.")
    model_g.add_arg("save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    model_g.add_arg("use_cuda", type=bool, default=True, help="Using CUDA when available")
    model_g.add_arg("overwrite_output_dir", type=bool, default=False, help="Overwrite the content of the output directory")
    model_g.add_arg("overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")
    model_g.add_arg("seed", type=int, default=42, help="random seed for initialization")
    model_g.add_arg("local_rank", type=int, default=-1, help="For distributed training: local_rank")
    model_g.add_arg("server_ip", type=str, default="", help="For distant debugging.")
    model_g.add_arg("server_port", type=str, default="", help="For distant debugging.")
    return parser
