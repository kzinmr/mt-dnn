# Copyright (c) Microsoft. All rights reserved.
import numpy as np
from sklearn.metrics import classification_report
from random import shuffle
from data_utils.metrics import Metric, METRIC_FUNC
from data_utils.task_def import DataFormat
from functools import reduce
import multiprocessing


def submit(path, data, label_dict=None):
    header = 'index\tprediction'
    with open(path ,'w') as writer:
        predictions, uids = data['predictions'], data['uids']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write('{}\t{}\n'.format(uid, pred))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\n'.format(uid, label_dict[pred]))


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
        if prev_bio == 'O' and bio == 'I' or prev_bio == 'B' and bio == 'I' and prev_netype != netype:
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
        if prev_bio == 'O' and bio == 'I' or prev_bio == 'B' and bio == 'I' and prev_netype != netype:
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
        results = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            results.append(p.map(_beam_search_decoder, args))
        label_ids_beams_list = results[0]
        label_ids_beams_list = [[beam[0] for beam in label_ids_beams] for label_ids_beams in label_ids_beams_list]
        labels_beams_list = [[__convert_labels(beam, id2label) for beam in label_ids_beams] for label_ids_beams in label_ids_beams_list]
        # validationを満たすbeamを選択; 無ければ先頭のbeam
        args = labels_beams_list
        label_ids_pred_idx = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            label_ids_pred_idx.append(p.map(_is_valid_labels, args))
        label_ids_pred_idx = label_ids_pred_idx[0]
        predictions_bs = [label_ids_beams_list[i][beam_idx] for (i, beam_idx) in enumerate(label_ids_pred_idx)]
        # validation(predictions)
        args = [__convert_labels(label_ids, id2label) for label_ids in predictions_bs]
        labels_fixed = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            labels_fixed.append(p.map(_remove_invalid_labels, args))
        labels_fixed = labels_fixed[0]
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
        if any(use_indices):
            if export_file is not None:
                with open(export_file, 'w', encoding='utf8') as f:
                    print(classification_report(
                np.array(_flatten_list(golds))[use_indices],
                np.array(_flatten_list(predictions))[use_indices],
                # labels=range(4, n_labels),
                # target_names=vocab.get_vocab_list()[4:]
                labels=range(4, n_labels),
                target_names=all_labels
                    ), file=f)
            print(classification_report(
                np.array(_flatten_list(golds))[use_indices],
                np.array(_flatten_list(predictions))[use_indices],
                # labels=range(4, n_labels),
                # target_names=vocab.get_vocab_list()[4:]
                labels=range(4, n_labels),
                target_names=all_labels
            ))
        for mm in metric_meta:
            metric_name = mm.name
            metric_func = METRIC_FUNC[mm]
            metric = metric_func(
                np.array(_flatten_list(predictions))[use_indices],
                np.array(_flatten_list(golds))[use_indices]
            )
            metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids, inputs

