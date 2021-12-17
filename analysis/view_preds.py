import argparse
import collections
import pandas as pd
import numpy as np
import json
import os

from tqdm import tqdm


class BadI(ValueError):
    pass


def read_gold(data, path, key='gold'):
    def read_tokens(path):
        corpus = []
        with open(path) as f:
            for line in f:
                corpus.append(line.strip().split())
        return corpus

    def read_tags(path):
        corpus = []
        with open(path) as f:
            for line in f:
                corpus.append(line.strip().split())
        return corpus

    d = {}
    d['tokens'] = read_tokens(os.path.join(path, 'seq.in'))
    d['tags'] = read_tags(os.path.join(path, 'seq.out'))

    data[key] = d

    return data


def read_pred(data, path, key='pred'):
    def parse(col):
        return [x.strip().split() for x in col]
    df = pd.read_csv(path)

    d = {}
    for k in ['slot_raw_pred', 'slot_processed_pred', 'slot_act']:
        d[k] = parse(df[k].tolist())
    d['slot_f1'] = df[k].tolist()

    data[key] = d

    return data


def check_spans(data):
    def get_spans(tags):
        start = 0
        spans = []

        while start < len(tags):
            t0 = tags[start]
            if t0 == 'O':
                start += 1
                continue

            elif t0.startswith('I'):
                raise BadI(tags)

            elif t0.startswith('B'):
                size_ = 1
                for size in range(2, len(tags)):
                    if start + size > len(tags):
                        break

                    t1 = tags[start + size - 1]

                    if t1.startswith('O') or t1.startswith('B'):
                        break

                    elif t1.startswith('I'):
                        size_ = size

                    else:
                        raise Exception(tags)

                spans.append((start, size_))
                start += size_
                continue

            else:
                raise Exception(tags)

        return spans

    m = collections.defaultdict(list)

    for p, g in zip(data['pred']['slot_processed_pred'], data['pred']['slot_act']):
        if 'UNK' in g:
            m['skipped-UNK'].append((p, g))
            continue

        assert len(p) == len(g)
        try:
            p_spans = set(get_spans(p))
        except BadI as e:
            m['skipped-I'].append((p, g))
            continue

        g_spans = set(get_spans(g))

        if len(g_spans) == 0:
            m['skipped-empty'].append((p, g))
            continue

        recall = len(set.intersection(g_spans, p_spans)) / len(g_spans)
        m['recall'].append(recall)

    print('skipped-UNK', len(m['skipped-UNK']))
    print('skipped-empty', len(m['skipped-empty']))
    print('skipped-I', len(m['skipped-I']))
    print('recall', np.mean(m['recall']), len(m['recall']))


def check_tags(data):

    c_tags = collections.Counter()
    for tags in data['ref']['tags']:
        for x in tags:
            c_tags[x] += 1

    for i, k in enumerate(sorted(c_tags.keys())):
        v = c_tags[k]
        print(i, k, c_tags[k])

    assert 'UNK' in c_tags


def main():
    data = read_gold({}, args.ref, key='ref')
    data = read_gold(data, args.gold, key='gold')
    data = read_pred(data, args.pred, key='pred')

    check_tags(data)
    check_spans(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='/mnt/nfs/scratch1/siddharthami/nlp/JointBERT/data/atis/test')
    parser.add_argument('--gold', default='/mnt/nfs/scratch1/siddharthami/nlp/JointBERT/data/atis.bracketed.ground_truth/test')
    parser.add_argument('--pred', default='/mnt/nfs/scratch1/siddharthami/nlp/JointBERT/models/atis.bracketed.ground_truth/seq.preds', type=str)
    args = parser.parse_args()
    main()

