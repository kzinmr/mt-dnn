# Copyright (c) Microsoft. All rights reserved.
import numpy as np
from random import shuffle
from data_utils.metrics import Metric, METRIC_FUNC
from data_utils.task_def import DataFormat
from functools import reduce
import multiprocessing
import os
from experiments.japanese.mytokenization import BertTokenizer


class ChunkEvaluation:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def extract_chunk(sentence, from_gold=True):
        chunks = []
        chunk = []
        for surf, pred, gold in sentence:
            if from_gold:
                if '-' in gold:
                    bio, netype = gold.split('-')
                    if bio == 'B':
                        chunk = [(surf, pred, gold)]
                    elif bio == 'I':
                        chunk.append((surf, pred, gold))
                elif chunk:
                    chunks.append(chunk)
                    chunk = []
            else:
                if '-' in pred:
                    bio, netype = pred.split('-')
                    if bio == 'B':
                        chunk = [(surf, pred, gold)]
                    elif bio == 'I':
                        chunk.append((surf, pred, gold))
                elif chunk:
                    chunks.append(chunk)
                    chunk = []
        if chunk:
            chunks.append(chunk)
        return chunks

    @staticmethod
    def __is_exact_match(chunk):
        return all(pred == gold for _, pred, gold in chunk)

    def check_chunks_match(self, chunks_pos, chunks_err_gold, chunks_err_pred):
        tp = 0
        for chunks in chunks_pos:
            for chunk in chunks:
                assert self.__is_exact_match(chunk)
                tp += 1
            # sentence
        for chunks in chunks_err_gold:
            for chunk in chunks:
                if self.__is_exact_match(chunk):
                    tp += 1

        fn = 0
        for chunks in chunks_err_gold:
            for chunk in chunks:
                if not self.__is_exact_match(chunk):
                    fn += 1

        fp = 0
        for chunks in chunks_err_pred:
            for chunk in chunks:
                if not self.__is_exact_match(chunk):
                    fp += 1

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 / (1 / p + 1 / r)
        return {'TP': tp, 'FP': fp, 'FN': fn, 'P': p, 'R': r, 'F1': f1}

    @staticmethod
    def __to_string(chunk):
        return '\n'.join('\t'.join(l) for l in chunk)

    def chunk_error_detail(self, chunks_err_gold, suffix='test'):
        fn_partial = 0  # 部分一致誤り
        fn_confusion = 0  # クラス誤り
        fn_o = 0  # 未抽出誤り

        part_list, o_list, conf_list = [], [], []
        for chunks in chunks_err_gold:
            for chunk in chunks:
                if any(pred.split('-')[-1] == gold.split('-')[-1] for _, pred, gold in chunk):
                    fn_partial += 1
                    part_list.append(self.__to_string(chunk))
                elif all(pred == 'O' for _, pred, gold in chunk):
                    fn_o += 1
                    o_list.append(self.__to_string(chunk))
                else:
                    fn_confusion += 1
                    conf_list.append(self.__to_string(chunk))
        with open(os.path.join(self.output_dir, f'chunk_error_partial_{suffix}.txt'), 'w', encoding='utf8') as f_partial:
            f_partial.write('\n\n'.join(part_list))
        with open(os.path.join(self.output_dir, f'chunk_error_confusion_{suffix}.txt'), 'w', encoding='utf8') as f_confusion:
            f_confusion.write('\n\n'.join(conf_list))
        with open(os.path.join(self.output_dir, f'chunk_error_o_{suffix}.txt'), 'w', encoding='utf8') as f_o:
            f_o.write('\n\n'.join(o_list))

        return {'partial': fn_partial, 'confusion': fn_confusion, 'O': fn_o}

    def chunkwise_evaluation(self, sentences, suffix='test'):
        sentences_err = [s for s in sentences if any(pred != gold for surf, pred, gold in s)]
        sentences_pos = [s for s in sentences if all(pred == gold for surf, pred, gold in s)]
        chunks_pos = [self.extract_chunk(sentence) for sentence in sentences_pos]
        chunks_err_gold = [self.extract_chunk(sentence) for sentence in sentences_err]
        chunks_err_pred = [self.extract_chunk(sentence, from_gold=False) for sentence in sentences_err]
        chunk_metric = self.check_chunks_match(chunks_pos, chunks_err_gold, chunks_err_pred)
        error_detail = self.chunk_error_detail(chunks_err_gold, suffix)
        return {'metric': chunk_metric, 'detail': error_detail}


class SubwordWordConverter:

    def __init__(self, bert_vocab, id2label, output_dir=None):
        self.tokenizer = BertTokenizer(bert_vocab, do_lower_case=False)
        self.id2label = id2label
        label2id = {v: k for k, v in id2label.items()}
        # self.LABELID_PAD = 0
        self.LABELID_CLS = label2id['[CLS]']
        self.LABELID_SEP = label2id['[SEP]']
        self.LABELID_X = label2id['X']
        self.ignore_label_ids = { # self.LABELID_PAD,
                                 self.LABELID_CLS, self.LABELID_SEP}
        # self.TOKENID_PAD = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.TOKENID_CLS = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.TOKENID_SEP = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.ignore_token_ids = {  #self.TOKENID_PAD,
                                 self.TOKENID_CLS, self.TOKENID_SEP}

        self.output_dir = output_dir

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

    def convert_ids_to_surfaces_list(self, token_ids_list, label_ids_list_pred, label_ids_list_gold, subword=False, suffix='test'):
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
                if self.output_dir is not None:
                    output_lines = [f'{word}\t{label}\t{label_gold}'
                                    for word, label, label_gold in zip(words, labels_pred, labels_gold)]
                    output_line = "\n".join(output_lines)
                    output_line += "\n\n"
                    output_sentences.append(output_line)
        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, f'token_label_pred_{suffix}.txt'), 'w', encoding='utf-8') as writer:
                for output_sentence in output_sentences:
                    writer.write(output_sentence)
        return [[(token, label_pred, label_gold)
                 for token, label_pred, label_gold in zip(tokens, labels_pred, labels_gold)]
                for tokens, labels_pred, labels_gold in zip(tokens_list, labels_list_pred, labels_list_gold)]


def _flatten_list(l):
    return reduce(lambda a, b: a + b, l)

def _beam_search_decoder(args):
    probability, k = args
    topk_candidates = [(list(), 1.0)]
    for row in probability:
        topk_candidates = sorted([(seq + [i], score * -np.log(prob))  # NOTE: prob==0.?
                                    for seq, score in topk_candidates
                                    for i, prob in enumerate(row)],
                                    key=lambda x: x[1])[:k]
    return topk_candidates

def __convert_labels(label_ids, id2label):
    return [id2label.get(i, '[NULL]') for i in label_ids]

def _is_valid(labels):
    prev_bio, prev_netype, netype = '', '', ''
    for l in labels:
        if len(l.split('-')) == 2:
            bio, netype = l.split('-')
        else:
            bio = l
        # check two bad patterns like ['O', 'I-A', 'O'] or ['B-A', 'I-B', 'O']
        if prev_bio in {'', 'O'} and bio == 'I' or prev_bio == 'B' and bio == 'I' and prev_netype != netype:
            return False
        prev_bio = bio
        prev_netype = netype
    return True

def _is_valid_labels(args):
    labels_beams = args
    beam_idx = 0
    for idx, labels in enumerate(labels_beams):
        if _is_valid(labels):
            beam_idx = idx
            break
    else:
        beam_idx = 0
    return beam_idx

def _remove_invalid_labels(labels):
    if _is_valid(labels) or len(labels) == 0:
        return labels

    prev_bio, prev_netype, netype = '', '', ''
    i = 0
    remove_indices = []
    seq_length = len(labels)
    while i < seq_length:
        l = labels[i]
        if l == 'X':
            i += 1
            continue
        if len(l.split('-')) == 2:
            bio, netype = l.split('-')
        else:
            bio = l
        # check two bad patterns like ['O', 'I-A', 'O'] or ['B-A', 'I-B', 'O']
        if prev_bio in {'', 'O'} and bio == 'I' or prev_bio == 'B' and bio == 'I' and prev_netype != netype:
            remove_from = i
            j = i + 1
            while j < seq_length and labels[j].split('-')[0] == 'I':
                j += 1
            remove_to = j
            remove_indices.append((remove_from, remove_to))
            i = j
        else:
            i += 1
        prev_bio = bio
        prev_netype = netype

    for (rm_from, rm_to) in remove_indices:
        labels[rm_from: rm_to] = ['O' for _ in range(rm_to - rm_from)]
    return labels

def eval_model(model, data, metric_meta, vocab, use_cuda=True, with_label=True, beam_search=True, beam_width=5, export_file=None):
    label2id = vocab.tok2ind
    id2label = vocab.ind2tok
    n_labels = len(label2id)
    all_labels = [id2label[i] for i in range(n_labels)]

    data.reset()
    if use_cuda:
        model.cuda()
    inputs = []
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        # batch_data: input, _, mask
        input_data = batch_data[0]  # (batch_size, batch_seq_length)
        score, pred, gold = model.predict(batch_meta, batch_data)
        batch_size = len(gold)
        true_seq_length = [len(g) for g in gold]
        batch_seq_length = int(len(pred) / batch_size)

        # Make (inputs_, preds, gold)'s shape as (batch_size, true_seq_length_i)
        inputs_ = [input_sentence[:t_l].cpu().detach().numpy().tolist() for input_sentence, t_l in zip(input_data, true_seq_length)]
        inputs.extend(inputs_)

        preds = [pred[i * batch_seq_length:(i * batch_seq_length + t_l)] for i, t_l in enumerate(true_seq_length)]
        predictions.extend(preds)

        score_ = [[score[(i * batch_seq_length + j) * n_labels:(i * batch_seq_length + j + 1) * n_labels] for j in range(t_l)] for i, t_l in enumerate(true_seq_length)]
        scores.extend(score_)

        golds.extend(gold)
        ids.extend(batch_meta['uids'])

    # (inputs, predictions, golds)'s shape: (n_data, true_seq_length_i)
    if beam_search:
        # print(id2label)
        # print([[id2label[g] for g in gs] for gs in golds])
        # print([[id2label[p] for p in ps] for ps in predictions])
        # print(len(scores), len(scores[0]), len(scores[0][1]))
        # beam_search(scores, beam_width):

        args = [(probability, beam_width) for probability in scores]
        label_ids_beams_list = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            label_ids_beams_list = p.map(_beam_search_decoder, args)

        label_ids_beams_list = [[beam[0] for beam in label_ids_beams] for label_ids_beams in label_ids_beams_list]
        labels_beams_list = [[__convert_labels(beam, id2label) for beam in label_ids_beams] for label_ids_beams in label_ids_beams_list]
        # validationを満たすbeamを選択; 無ければ先頭のbeam
        args = labels_beams_list
        label_ids_pred_idx = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            label_ids_pred_idx = p.map(_is_valid_labels, args)

        predictions_bs = [label_ids_beams_list[i][beam_idx] for (i, beam_idx) in enumerate(label_ids_pred_idx)]
        # validation(predictions)
        args = [__convert_labels(label_ids, id2label) for label_ids in predictions_bs]
        labels_fixed = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            labels_fixed = p.map(_remove_invalid_labels, args)

        predictions = [[label2id.get(l, n_labels) for l in labels] for labels in labels_fixed]
        scores = [ss for ss in scores for s in ss]

        # print([[id2label[p] for p in ps] for ps in predictions])

    # remove ["O", "X", "[CLS]", "[SEP]"] from evaluation
    # all task must add labels in the order below
    # LabelMapper.add("X")
    # LabelMapper.add("[CLS]")
    # LabelMapper.add("[SEP]")
    # LabelMapper.add("O")

    use_indices = [label > 3 for label in _flatten_list(golds)]
    if with_label:
        for mm in metric_meta:
            metric_name = mm.name
            metric_func = METRIC_FUNC[mm]
            metric = metric_func(
                np.array(_flatten_list(predictions))[use_indices],
                np.array(_flatten_list(golds))[use_indices]
            )
            metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids, inputs
