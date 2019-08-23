# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertConfig
from experiments.exp_def import TaskDefs
# from experiments.glue.glue_utils import submit, eval_model
from experiments.japanese.bccwj_utils import submit, eval_model
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from data_utils.task_def import TaskType
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel
from experiments.japanese.mytokenization import BertTokenizer



class SubwordWordConverter:

    def __init__(self, tokenizer, id2label, export_file=None):
        self.tokenizer = tokenizer
        self.id2label = id2label
        label2id = {v: k for k, v in id2label.items()}
        # self.LABELID_PAD = 0
        self.LABELID_CLS = label2id['[CLS]']
        self.LABELID_SEP = label2id['[SEP]']
        self.LABELID_X = label2id['X']
        self.ignore_label_ids = { # self.LABELID_PAD,
                                 self.LABELID_CLS, self.LABELID_SEP}
        # self.TOKENID_PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.TOKENID_CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.TOKENID_SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.ignore_token_ids = {  #self.TOKENID_PAD,
                                 self.TOKENID_CLS, self.TOKENID_SEP}

        self.export_file = export_file

    @staticmethod
    def convert_subword_to_word_by_label(subwords, labels_gold):
        # subword.startswith('##') == True だけがsubwordとは限らない
        # 'X' label を subword　-> word の復元に用いる
        words, labels = [], []
        for sw, lb in zip(subwords, labels_gold):
            if lb == 'X':
                assert len(words) > 0
                prev = words[-1]
                words = words[:-1]
                word = prev + sw[2:]  # '##' 以降
            else:
                word = sw
                labels.append(lb)
            words.append(word)
        return words, labels

    def check_separator_aligned(self, inputs, labels):
        for i, label in zip(inputs, labels):
            if label == self.LABELID_CLS:
                if i != self.TOKENID_CLS:
                    return False
            elif label == self.LABELID_SEP:
                if i != self.TOKENID_SEP:
                    return False
        return True

    def filter_token_ids(self, token_ids):
        return [i for i in token_ids if i not in self.ignore_token_ids]

    def filter_label_ids(self, label_ids):
        return [l for l in label_ids if l not in self.ignore_label_ids]

    def convert_id_to_surface_token(self, token_ids):
        token_ids = self.filter_token_ids(token_ids)
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def convert_id_to_surface_label(self, label_ids):
        label_ids = self.filter_label_ids(label_ids)
        return [self.id2label[i] for i in label_ids]

    def convert_tokens_to_words(self, token_ids, label_ids_gold, subword=False):
        # subword　-> word の復元
        subwords = self.convert_id_to_surface_token(token_ids)
        labels = self.convert_id_to_surface_label(label_ids_gold)
        if subword:
            return subwords, labels
        else:
            words, labels = self.convert_subword_to_word_by_label(
                zip(subwords, labels))
            return words, labels

    def filter_label_ids_by_gold(self, label_ids_pred, label_ids_gold):
        label_ids_pred = self.filter_label_ids(label_ids_pred)
        label_ids_gold = self.filter_label_ids(label_ids_gold)
        label_ids_pred = [l for l, lg in zip(label_ids_pred, label_ids_gold)
                          if lg != self.LABELID_X]
        return label_ids_pred

    def convert_ids_to_surfaces(self, token_ids, label_ids_pred, label_ids_gold, subword=False):
        # gold label が 'X' であるか否かを基点に subword かどうかを認識する
        subwords = self.convert_id_to_surface_token(token_ids)
        labels_gold = self.convert_id_to_surface_label(label_ids_gold)
        if subword:
            words = subwords
        else:
            # subwords => words
            words, labels_gold = self.convert_subword_to_word_by_label(
                subwords, labels_gold)
            # subword_labels => word_labels
            label_ids_pred = self.filter_label_ids_by_gold(
                label_ids_pred, label_ids_gold)
        labels_pred = self.convert_id_to_surface_label(label_ids_pred)

        return words, labels_pred, labels_gold

    def convert_ids_to_surfaces_list(self, token_ids_list, label_ids_list_pred, label_ids_list_gold, subword=False):
        output_sentences = []
        tokens_list, labels_list_pred, labels_list_gold = [], [], []
        for token_ids, label_ids_pred, label_ids_gold in zip(token_ids_list, label_ids_list_pred, label_ids_list_gold):
            if self.check_separator_aligned(token_ids, label_ids_pred):
                words, labels_pred, labels_gold = self.convert_ids_to_surfaces(
                    token_ids, label_ids_pred, label_ids_gold, subword=subword)
                tokens_list.append(words)
                labels_list_pred.append(labels_pred)
                labels_list_gold.append(labels_gold)

                # export
                if self.export_file is not None:
                    output_lines = [f'{word}\t{label}\t{label_gold}'
                                    for word, label, label_gold in zip(words, labels_pred, labels_gold)]
                    output_line = "\n".join(output_lines)
                    output_line += "\n\n"
                    output_sentences.append(output_line)
        if self.export_file is not None:
            with open(self.export_file, 'w', encoding='utf-8') as writer:
                for output_sentence in output_sentences:
                    writer.write(output_sentence)

        return tokens_list, labels_list_pred, labels_list_gold


def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1,2')
    parser.add_argument('--label_size', type=str, default='3')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--init_ratio', type=float, default=1)
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument("--init_checkpoint", default='mt_dnn_models/bert_model_base.pt', type=str)
    parser.add_argument("--bert_config_path", default='mt_dnn_models/bert_model_base.pt', type=str)
    parser.add_argument('--bert_vocab', type=str, default='mt_dnn_models/vocab.txt')
    parser.add_argument('--data_dir', default='data/canonical_data/mt_dnn_uncased_lower')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml")
    parser.add_argument('--train_datasets', default='mnli')
    parser.add_argument('--test_datasets', default='mnli_mismatched,mnli_matched')
    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    parser.add_argument("--model_ckpt", default='checkpoints/model_0.pt', type=str)
    parser.add_argument("--resume", action='store_true')

    # EMA
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--ema_gamma', type=float, default=0.995)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)
    return parser


parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir
args.train_datasets = args.train_datasets.split(',')
args.test_datasets = args.test_datasets.split(',')
pprint(args)

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)
logger.info(args.answer_opt)

task_defs = TaskDefs(args.task_def)


def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def generate_decoder_opt(enable_san, max_opt):
    opt_v = 0
    if enable_san and max_opt < 3:
        opt_v = max_opt
    return opt_v

def main():
    logger.info('Launching the MT-DNN evaluation')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size
    # train_data_list = []
    tasks = {}
    tasks_class = {}
    nclass_list = []
    decoder_opts = []
    dropout_list = []

    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks: continue
        assert prefix in task_defs.n_class_map
        assert prefix in task_defs.data_type_map
        # data_type = task_defs.data_type_map[prefix]
        nclass = task_defs.n_class_map[prefix]
        task_id = len(tasks)
        if args.mtl_opt > 0:
            task_id = tasks_class[nclass] if nclass in tasks_class else len(tasks_class)

    #     task_type = task_defs.task_type_map[prefix]
    #     pw_task = False
    #     if task_type == TaskType.Ranking:
    #         pw_task = True

        # dopt = generate_decoder_opt(task_defs.enable_san_map[prefix], opt['answer_opt'])
        dopt = task_defs.enable_san_map[prefix]
        if task_id < len(decoder_opts):
            decoder_opts[task_id] = min(decoder_opts[task_id], dopt)
        else:
            decoder_opts.append(dopt)

        if prefix not in tasks:
            tasks[prefix] = len(tasks)
            if args.mtl_opt < 1: nclass_list.append(nclass)

        if (nclass not in tasks_class):
            tasks_class[nclass] = len(tasks_class)
            if args.mtl_opt > 0: nclass_list.append(nclass)

        dropout_p = task_defs.dropout_p_map.get(prefix, args.dropout_p)
        dropout_list.append(dropout_p)

    #     train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
    #     logger.info('Loading {} as task {}'.format(train_path, task_id))
    #     train_data = BatchGen(BatchGen.load(train_path, True, pairwise=pw_task, maxlen=args.max_seq_len),
    #                           batch_size=batch_size,
    #                           dropout_w=args.dropout_w,
    #                           gpu=args.cuda,
    #                           task_id=task_id,
    #                           maxlen=args.max_seq_len,
    #                           pairwise=pw_task,
    #                           data_type=data_type,
    #                           task_type=task_type)
    #     train_data_list.append(train_data)

    opt['answer_opt'] = decoder_opts
    opt['tasks_dropout_p'] = dropout_list

    args.label_size = ','.join([str(l) for l in nclass_list])
    logger.info(args.label_size)
    dev_data_list = []
    test_data_list = []
    for dataset in args.test_datasets:
        prefix = dataset.split('_')[0]
        task_id = tasks_class[task_defs.n_class_map[prefix]] if args.mtl_opt > 0 else tasks[prefix]
        task_type = task_defs.task_type_map[prefix]

        pw_task = False
        if task_type == TaskType.Ranking:
            pw_task = True

        assert prefix in task_defs.data_type_map
        data_type = task_defs.data_type_map[prefix]

        dev_path = os.path.join(data_dir, '{}_dev.json'.format(dataset))
        dev_data = None
        if os.path.exists(dev_path):
            dev_data = BatchGen(BatchGen.load(dev_path, False, pairwise=pw_task, maxlen=args.max_seq_len),
                                batch_size=args.batch_size_eval,
                                gpu=args.cuda, is_train=False,
                                task_id=task_id,
                                maxlen=args.max_seq_len,
                                pairwise=pw_task,
                                data_type=data_type,
                                task_type=task_type)
        dev_data_list.append(dev_data)

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data = BatchGen(BatchGen.load(test_path, False, pairwise=pw_task, maxlen=args.max_seq_len),
                                 batch_size=args.batch_size_eval,
                                 gpu=args.cuda, is_train=False,
                                 task_id=task_id,
                                 maxlen=args.max_seq_len,
                                 pairwise=pw_task,
                                 data_type=data_type,
                                 task_type=task_type)
        test_data_list.append(test_data)

    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)

    # all_iters = [iter(item) for item in train_data_list]
    # all_lens = [len(bg) for bg in train_data_list]

    # # div number of grad accumulation. 
    # num_all_batches = args.epochs * sum(all_lens) // args.grad_accumulation_step
    # logger.info('############# Gradient Accumulation Infor #############')
    # logger.info('number of step: {}'.format(args.epochs * sum(all_lens)))
    # logger.info('number of grad grad_accumulation step: {}'.format(args.grad_accumulation_step))
    # logger.info('adjusted number of step: {}'.format(num_all_batches))

    # if len(train_data_list) > 1 and args.ratio > 0:
    #     num_all_batches = int(args.epochs * (len(train_data_list[0]) * (1 + args.ratio)))

    bert_config_path = args.bert_config_path
    state_dict = None

    if os.path.exists(bert_config_path):
        config = json.load(open(bert_config_path, encoding="utf-8"))
        config['attention_probs_dropout_prob'] = args.bert_dropout_p
        config['hidden_dropout_prob'] = args.bert_dropout_p
        opt.update(config)
    else:
        logger.error('#' * 20)
        logger.error('Could not find the init model!\n The parameters will be initialized randomly!')
        logger.error('#' * 20)
        config = BertConfig(vocab_size_or_config_json_file=30522).to_dict()
        opt.update(config)

    model = MTDNNModel(opt, bert_init_checkpoint=args.init_checkpoint, state_dict=state_dict)
    assert args.model_ckpt
    logger.info('loading model from {}'.format(args.model_ckpt))
    model.load(args.model_ckpt)

    # #### model meta str
    # headline = '############# Model Arch of MT-DNN #############'
    # ### print network
    # logger.info('\n{}\n{}\n'.format(headline, model.network))

    # # dump config
    # config_file = os.path.join(output_dir, 'config.json')
    # with open(config_file, 'w', encoding='utf-8') as writer:
    #     writer.write('{}\n'.format(json.dumps(opt)))
    #     writer.write('\n{}\n{}\n'.format(headline, model.network))

    # logger.info("Total number of params: {}".format(model.total_param))

    if args.freeze_layers > 0:
        model.network.freeze_layers(args.freeze_layers)

    # for epoch in range(0, args.epochs):
    epoch = os.path.basename(args.model_ckpt).split('.')[0].split('_')[-1]
    tokenizer = BertTokenizer(args.bert_vocab, do_lower_case=False)
    if True:
        # logger.warning('At epoch {}'.format(epoch))
        # for train_data in train_data_list:
        #     train_data.reset()
        # start = datetime.now()
        # all_indices = []
        # if len(train_data_list) > 1 and args.ratio > 0:
        #     main_indices = [0] * len(train_data_list[0])
        #     extra_indices = []
        #     for i in range(1, len(train_data_list)):
        #         extra_indices += [i] * len(train_data_list[i])
        #     random_picks = int(min(len(train_data_list[0]) * args.ratio, len(extra_indices)))
        #     extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
        #     if args.mix_opt > 0:
        #         extra_indices = extra_indices.tolist()
        #         random.shuffle(extra_indices)
        #         all_indices = extra_indices + main_indices
        #     else:
        #         all_indices = main_indices + extra_indices.tolist()

        # else:
        #     for i in range(1, len(train_data_list)):
        #         all_indices += [i] * len(train_data_list[i])
        #     if args.mix_opt > 0:
        #         random.shuffle(all_indices)
        #     all_indices += [0] * len(train_data_list[0])
        # if args.mix_opt < 1:
        #     random.shuffle(all_indices)

        # for i in range(len(all_indices)):
        #     task_id = all_indices[i]
        #     batch_meta, batch_data = next(all_iters[task_id])
        #     model.update(batch_meta, batch_data)
        #     if (model.local_updates) % (args.log_per_updates * args.grad_accumulation_step) == 0 or model.local_updates == 1:
        #         ramaining_time = str((datetime.now() - start) / (i + 1) * (len(all_indices) - i - 1)).split('.')[0]
        #         logger.info('Task [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(task_id,
        #                                                                                             model.updates,
        #                                                                                             model.train_loss.avg,
        #                                                                                             ramaining_time))

        #     if args.save_per_updates_on and ((model.local_updates) % (args.save_per_updates * args.grad_accumulation_step) == 0):
        #         model_file = os.path.join(output_dir, 'model_{}_{}.pt'.format(epoch, model.updates))
        #         logger.info('Saving mt-dnn model to {}'.format(model_file))
        #         model.save(model_file)

        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            label_dict = task_defs.global_map.get(prefix, None)
            dev_data = dev_data_list[idx]
            if dev_data is not None:
                classification_report_file = os.path.join(output_dir, '{}_dev_classification_report_{}.json'.format(dataset, epoch))
                dev_metrics, dev_predictions, scores, golds, dev_ids, dev_inputs = eval_model(model, dev_data, task_defs.metric_meta_map[prefix], label_dict,
                                                                                 use_cuda=args.cuda,
                                                                                 export_file=classification_report_file)
                for key, val in dev_metrics.items():
                    logger.warning("Task {0} -- epoch {1} -- Dev {2}: {3:.3f}".format(dataset, epoch, key, val))
                metric_file = os.path.join(output_dir, '{}_dev_metrics_{}.json'.format(dataset, epoch))
                dump(metric_file, dev_metrics)

                score_file = os.path.join(output_dir, '{}_dev_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': dev_metrics, 'predictions': dev_predictions, 'uids': dev_ids, 'scores': scores}
                dump(score_file, results)
                # official_score_file = os.path.join(output_dir, '{}_dev_scores_{}.tsv'.format(dataset, epoch))
                # submit(official_score_file, results, label_dict)

                export_file = os.path.join(output_dir, '{}_dev_token_label_pred_{}.txt'.format(dataset, epoch))
                swc = SubwordWordConverter(tokenizer, label_dict.ind2tok, export_file)
                token_ids = [inputs.cpu().detach().numpy() for inputs_list in dev_inputs for inputs in inputs_list]
                tokens_list, labels_list_pred, labels_list_gold = swc.convert_ids_to_surfaces_list(token_ids, dev_predictions, golds)
                # chunk-wise evaluation


            # test eval
            test_data = test_data_list[idx]
            if test_data is not None:
                classification_report_file = os.path.join(output_dir, '{}_test_classification_report_{}.json'.format(dataset, epoch))
                test_metrics, test_predictions, scores, golds, test_ids, test_inputs = eval_model(model, test_data,
                                                                                    task_defs.metric_meta_map[prefix],
                                                                                    label_dict,
                                                                                    use_cuda=args.cuda, with_label=True,
                                                                                    export_file=classification_report_file)
                for key, val in test_metrics.items():
                    logger.warning("Task {0} -- epoch {1} -- Test {2}: {3:.3f}".format(dataset, epoch, key, val))
                metric_file = os.path.join(output_dir, '{}_test_metrics_{}.json'.format(dataset, epoch))
                dump(metric_file, test_metrics)

                score_file = os.path.join(output_dir, '{}_test_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
                dump(score_file, results)
                # official_score_file = os.path.join(output_dir, '{}_test_scores_{}.tsv'.format(dataset, epoch))
                # submit(official_score_file, results, label_dict)
                # logger.info('[new test scores saved.]')
                export_file = os.path.join(output_dir, '{}_test_token_label_pred_{}.txt'.format(dataset, epoch))
                swc = SubwordWordConverter(tokenizer, label_dict.ind2tok, export_file)
                token_ids = [inputs.cpu().detach().numpy() for inputs_list in test_inputs for inputs in inputs_list]

                # sentences = [
                #     [(token, label_pred, label_gold)
                #      for token, label_pred, label_gold in zip(tokens, labels_pred, labels_gold)]
                #     for tokens, labels_pred, labels_gold in zip(token_ids, test_predictions, golds)]
                # sentences_prev = [sentence for sentence in sentences if any(x[-1] != 'O' for x in sentence)]

                tokens_list, labels_list_pred, labels_list_gold = swc.convert_ids_to_surfaces_list(token_ids, test_predictions, golds)

                # sentences = [
                #     [(token, label_pred, label_gold)
                #      for token, label_pred, label_gold in zip(tokens, labels_pred, labels_gold)]
                #     for tokens, labels_pred, labels_gold in zip(tokens_list, labels_list_pred, labels_list_gold)]
                # sentences_p = [sentence for sentence in sentences if any(x[-1] != 'O' for x in sentence)]

                # chunk-wise evaluation

        # model_file = os.path.join(output_dir, 'model_{}.pt'.format(epoch))
        # model.save(model_file)


if __name__ == '__main__':
    main()


