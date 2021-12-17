import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert


MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer),
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1',
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    special_tokens = []
    with open(f'{args.data_dir}/{args.task}/{args.special_token_label_file}') as f:
        for token in f:
            special_tokens.append(token.strip())
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def post_process_slot_labels(slot_preds, slot_labels):
    brac, other = 'BRAC', 'O'
    proc_slot_preds = [ [ slot_p[ind] for ind in range(len(slot_l)) if slot_l[ind] != brac] for slot_p, slot_l in zip(slot_preds, slot_labels)]
    proc_slot_preds = [[sl if sl != brac else other for sl in slot] for slot in proc_slot_preds] 
    proc_slot_labels = [ [sl for sl in slot_l if sl != brac] for slot_l in slot_labels]
    return proc_slot_preds, proc_slot_labels

def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result, slot_f1_scores = get_slot_metrics(slot_preds, slot_labels)
    sementic_result, sem_results = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results, slot_f1_scores, sem_results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }, [f1_score([l], [p]) for l, p in zip(labels, preds)] 


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "semantic_frame_acc": semantic_acc
    }, (intent_result & slot_result).astype(int)
