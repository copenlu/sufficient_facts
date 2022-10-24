import argparse
import csv
import itertools
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from functools import partial
from string import punctuation
from typing import Dict

import numpy as np
import spacy
import torch
from modeling.data_reader import collate_explanations, get_datasets, \
    ArrayDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizerFast, RobertaForSequenceClassification, RobertaConfig, \
    RobertaTokenizerFast, AlbertForSequenceClassification, AlbertConfig, \
    AlbertTokenizerFast

MAX_TRAIN = 900
MAX_DEV = 200

FEVER_IDS = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
HOVER_IDS = {1: 'NOT_SUPPORTED', 0: 'SUPPORTED'}

field_names = ['claim', 'evidence', 'label']
field_names_extended = ['claim', 'evidence', 'label', 'id',
                        'adv_pred', 'orig_pred', 'orig_pred_logit', 'adv_pred_logit',
                        'orig_gold_logit', 'adv_gold_logit',
                        'type', 'tag']
regex_date_v1 = re.compile(
    r"(?P<date> (?P<month>January|February|March|April|May|June|July|August"
    r"|September|November|October|December)\ (?P<day>[1-9][0-9]?)\ \,"
    r"?\ (?P<year>[1-9][0-9]?[0-9]?[0-9]?))",
    re.VERBOSE)
regex_date_v2 = re.compile(
    r"(?P<date>(?P<day>[1-9][0-9]?)\ ("
    r"?P<month>January|February|March|April|May|June|July|August|September"
    r"|November|October|December)\ (?P<year>[1-9][0-9]?[0-9]?[0-9]?))",
    re.VERBOSE)


def get_predictions(args, models, tokenizers, collate_fns, dataset, prefix='adv'):
    for i, (model, tokenizer, collate_fn) in enumerate(
            zip(models, tokenizers, collate_fns)):
        logits, preds, gold = [], [], []
        softmax = torch.nn.Softmax(dim=1)
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate_fn,
                                 shuffle=False)

        for batch_i, batch in tqdm(enumerate(data_loader)):
            logits_instance = model(batch['input_ids_tensor'],
                                    attention_mask=batch[
                                                       'input_ids_tensor'] !=
                                                   tokenizer.pad_token_id).logits
            logits_instance = softmax(logits_instance).detach()
            preds += torch.argmax(logits_instance,
                                  dim=-1).cpu().numpy().tolist()
            logits += logits_instance.cpu().numpy().tolist()
            gold += batch['target_labels_tensor'].cpu().numpy().tolist()

        logit_gold = [instance_logits[gold[i]] for i, instance_logits in
                      enumerate(logits)]
        logit_pred = [instance_logits[preds[i]] for i, instance_logits in
                      enumerate(logits)]

        for instance_i in range(len(dataset)):
            if i == 0:
                dataset.dataset[instance_i][f'{prefix}_gold_logit'] = [
                    logit_gold[instance_i]]
                dataset.dataset[instance_i][f'{prefix}_pred_logit'] = [
                    logit_pred[instance_i]]
                dataset.dataset[instance_i][f'{prefix}_logits'] = [
                    logits[instance_i]]
                dataset.dataset[instance_i][f'{prefix}_pred'] = [
                    preds[instance_i]]
            else:
                dataset.dataset[instance_i][f'{prefix}_gold_logit'] += [
                    logit_gold[instance_i]]
                dataset.dataset[instance_i][f'{prefix}_pred_logit'] += [
                    logit_pred[instance_i]]
                dataset.dataset[instance_i][f'{prefix}_logits'] += [
                    logits[instance_i]]
                dataset.dataset[instance_i][f'{prefix}_pred'] += [
                    preds[instance_i]]

    return dataset


def remove_text(instance, sentence_i, tag, removed_text, new_text=None):
    new_instance = deepcopy(instance)
    new_instance['orig_text'] = deepcopy(new_instance['text'])

    num_occurrences = sum(new_instance['text'][sentence_i][1][
                          i:i + len(removed_text)] == removed_text for i in
                          range(len(new_instance['text'][sentence_i][1])))

    if num_occurrences != 1:
        return None
    if removed_text.strip().lower().startswith('of '):
        return None
    removed_text = removed_text.strip()
    new_instance['tag'] = tag
    new_instance['mturk_text'] = deepcopy(new_instance['orig_text'])

    new_instance['removed'] = removed_text if removed_text else new_text

    if not new_text:
        new_instance['text'][sentence_i] = (
            new_instance['text'][sentence_i][0],
            new_instance['text'][sentence_i][1].replace(removed_text, '').replace('  ', ' '))
        new_instance['mturk_text'][sentence_i] = (
            new_instance['mturk_text'][sentence_i][0],
            new_instance['mturk_text'][sentence_i][1].replace(
                removed_text,
                f'<span style="color:red;">{removed_text}</span>'))

    return new_instance


def format_mturk_text(mturk_sentences):
    mturk_text = ' '.join(
        [f'~=={title}==~ {text}' for title, text in mturk_sentences])
    mturk_text = mturk_text.replace('-LRB-', '(')
    mturk_text = mturk_text.replace('-LSB-', '[')
    mturk_text = mturk_text.replace('-RSB-', ']')
    mturk_text = mturk_text.replace('-RRB-', ')')
    mturk_text = mturk_text.replace('[[', '')
    mturk_text = mturk_text.replace(']]', '')
    mturk_text = mturk_text.replace('~==', '[[')
    mturk_text = mturk_text.replace('==~', ']]')
    mturk_text = mturk_text.replace('[[]]', '')
    mturk_text = mturk_text.replace('  ', '')

    return mturk_text


def write_to_files(prefix, instances):
    with open(f'{prefix}.csv', 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=field_names)
        writer.writeheader()
        for row in instances:
            writer.writerow({k: row[k.lower()] for k in field_names})

    with open(f'{prefix}_extended.csv', 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=field_names_extended)
        writer.writeheader()
        for row in instances:
            writer.writerow({k: row[k.lower()] for k in field_names_extended})


def get_perturbations(args, models, tokenizers, collate_fns, dataset, split) -> (Dict, Dict):
    model.eval()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    perturbations = []
    max_adv = MAX_TRAIN if split == 'train' else MAX_DEV
    NEI_LABEL = 1 if args.dataset == 'hover' else 2
    IDS_LABELS = HOVER_IDS if args.dataset == 'hover' else FEVER_IDS

    dataset = get_predictions(args, models, tokenizers, collate_fns, dataset,
                              prefix='orig')

    for instance_i, instance in enumerate(tqdm(dataset)):
        if any([instance['orig_pred'][j] != instance['label'] for j in range(3)]) \
                or instance['label'] == NEI_LABEL:
            continue

        for sent_i, sent in enumerate(instance['text']):
            sent_1 = sent[1].replace('-LRB- ', '(').replace('-LSB- ', '[').replace(' -RSB-', ']').replace(' -RRB-', ')')
            used_pps = []

            try:
                doc = nlp(sent_1)
            except Exception as e:
                print(e)
                continue

            parse = list(doc.sents)
            if not parse:
                continue
            parse = parse[0]

            perturbations += date_modifiers(instance, sent, sent_i)
            perturbations += noun_modifiers(instance, parse, sent_i)
            perturbations += adjective_modifiers(instance, parse, sent_i)
            perturbations += adverb_modifiers(instance, parse, sent_i)

            queue = [(parse, False)]
            while queue:
                item, pp_used = queue.pop(0)

                tags = [ch[0].tag_ for ch in item._.children if
                        len(ch._.labels) == 0]

                all_children = list(item._.children)

                if len(item._.labels) > 0:
                    pp_perturb = pp_adjunct(all_children, instance, sent_i)
                    pp_perturb_fitered = []
                    for pp in pp_perturb:
                        if any(pp['removed'] in used_pp for used_pp in used_pps):
                            continue
                        else:
                            pp_perturb_fitered.append(pp)
                            used_pps.append(pp['removed'])

                    perturbations += pp_perturb_fitered

                    perturbations += number_modifiers(all_children, instance, sent_i, tags, item)
                    perturbations += sbar(all_children, instance, item, sent, sent_i)

                    for const in item._.children:
                        queue.append((const, pp_used))

    dataset_adv = ArrayDataset(perturbations)
    dataset_adv = get_predictions(args, models, tokenizers, collate_fns, dataset_adv,
                                  prefix='adv')

    instance2advs = defaultdict(lambda: [])
    for adv in dataset_adv:
        instance2advs[adv['id']] += [adv]

    cursor, used_claims, used_claims_ds_id = [], set(), set()

    for instance_id, adversaries in instance2advs.items():
        adversaries_nei = [adv for adv in adversaries if all(
            adv['adv_pred'][i] == NEI_LABEL for i in range(len(models)))]
        if adversaries_nei:
            for adversary in adversaries_nei:
                if args.dataset != 'hover':
                    if adversary['id'] in used_claims or adversary[
                        'ds_id'] in used_claims_ds_id:
                        continue

                for_mturk = write_adv(IDS_LABELS, adversary,
                                      'all_nei')
                cursor.append(for_mturk)
                used_claims.add(adversary['id'])
                used_claims_ds_id.add(adversary['ds_id'])

        if len(cursor) >= max_adv:
            break

    write_to_files(f'ann/{args.dataset}_const/{split}_nei', cursor)
    cursor = []

    for instance_id, adversaries in instance2advs.items():
        adversaries_disagree = [adv for adv in adversaries if len(
            set(adv['adv_pred'])) == 2 and NEI_LABEL in set(adv['adv_pred']) and NEI_LABEL != adv['label'] and
                                adv['label'] in set(adv['adv_pred'])]

        if adversaries_disagree:
            adversary = min(adversaries_disagree,
                            key=lambda x: np.mean(x['adv_pred_logit']))

            if adversary['id'] in used_claims or adversary[
                'ds_id'] in used_claims_ds_id:
                continue
            for_mturk = write_adv(IDS_LABELS, adversary,
                                  'disagree')
            cursor.append(for_mturk)
            used_claims.add(adversary['id'])
            used_claims_ds_id.add(adversary['ds_id'])

        if len(cursor) >= max_adv:
            break

    write_to_files(f'ann/{args.dataset}_const/{split}_disagree', cursor)
    cursor = []

    for instance_id, adversaries in instance2advs.items():
        adversaries_ei = [adv for adv in adversaries if all(
            adv['adv_pred'][i] == adv['label'] and adv['label'] != NEI_LABEL for
            i in range(len(models)))]

        if adversaries_ei:
            adversary = min(adversaries_ei,
                            key=lambda x: np.mean(x['adv_pred_logit']))
            if adversary['id'] in used_claims or adversary[
                'ds_id'] in used_claims_ds_id:
                continue
            for_mturk = write_adv(IDS_LABELS, adversary, 'agree_ei')
            cursor.append(for_mturk)
            used_claims.add(adversary['id'])
            used_claims_ds_id.add(adversary['ds_id'])

        if len(cursor) >= max_adv:
            break

    write_to_files(f'ann/{args.dataset}_const/{split}_ei', cursor)
    cursor = []


def pp_adjunct(all_children, instance, sent_i):
    """Finds prepositional phrases."""
    perturbations = []
    adjunct_adverbials = []
    for i, (x1, x2) in enumerate(zip(all_children[:-1], all_children[1:])):
        if len(x1._.labels) > 0 and len(x2._.labels) > 0:
            if x2._.labels[0] == 'PP':
                if x1._.labels[0] == 'PP':
                    adjunct_adverbials.append((i + 1, x2.text))
                elif x1._.labels[0] == 'NP':
                    if 'NNP' in [token.tag_ for token in x1] and 'NNP' in [token.tag_ for token in x2]:
                        continue
                    else:
                        adjunct_adverbials.append((i + 1, x2.text))

    for idx, adjunct_adverbial in adjunct_adverbials:
        adjunct_child = all_children[idx]

        if idx > 0 and all_children[idx - 1].text == ',':
            adjunct_adverbial = f', {adjunct_adverbial}'
        if idx < len(all_children) - 1 and all_children[idx + 1].text == ',':
            adjunct_adverbial = f'{adjunct_adverbial} ,'

        type = f'PP'
        nents = set([token.ent_type_ for token in adjunct_child])
        if len(nents) > 1:
            nent = [ne for ne in nents if ne != ''][0]
            # type = type + '-' + nent

        new_instance = remove_text(instance, sent_i,
                                   type,
                                   removed_text=adjunct_adverbial)
        if new_instance:
            perturbations.append(new_instance)
    return perturbations


def number_modifiers(all_children, instance, sent_i, tags, item):
    """Finds number modifiers"""
    perturbations = []
    if item._.labels[0] == 'NP' and 'CD' in tags:
        for child in all_children:
            token = child[0]
            if token.tag_ == 'CD' and token.head.pos_ == 'NOUN' and \
                    token.head.ent_type_ != 'DATE':
                new_instance = remove_text(instance, sent_i,
                                           f'Num-mod-{token.ent_type_}',
                                           removed_text=token.text)
                if new_instance:
                    perturbations.append(new_instance)

    return perturbations


def sbar(all_children, instance, item, sent, sent_i):
    """Finds subordinate clauses."""
    perturbations = []
    if item._.labels[0] == 'SBAR':
        if (item[0].text == 'that' and item[0].tag_ == 'IN') or item[0].pos_ == 'SCONJ':
            pass
        else:
            remove_text_ = item.text
            if remove_text_ + ' ,' in sent[1]:
                remove_text_ = remove_text_ + ' ,'
            if ', ' + remove_text_ in sent[1]:
                remove_text_ = ', ' + remove_text_

            new_instance = remove_text(instance, sent_i,
                                       all_children[0]._.labels[0] if all_children and len(
                                           all_children[0]._.labels) > 0 else 'SBAR',
                                       removed_text=remove_text_)
            if new_instance:
                perturbations.append(new_instance)

    return perturbations


def adverb_modifiers(instance, parse, sent_i):
    """Finds adverb modifiers"""
    perturbations = []
    i = 0
    tokens_parse = list(parse)
    while i < len(tokens_parse):
        tokens = ''
        token = tokens_parse[i]

        while token.pos_ == 'ADV' and token.head.pos_ == 'VERB' and \
                token.dep_ in [
            'advmod', 'advcl'] and token.tag_ == 'RB':
            if i + 1 < len(tokens_parse) and tokens_parse[
                i + 1].text in punctuation:
                tokens = ''
                break
            tokens = tokens + ' ' + token.text
            i += 1
            if i == len(tokens_parse):
                break
            token = tokens_parse[i]

        if tokens:
            new_instance = remove_text(instance, sent_i,
                                       'Mod-Adv-VERB',
                                       removed_text=tokens)
            if new_instance:
                perturbations.append(new_instance)

        i += 1

    return perturbations


def adjective_modifiers(instance, parse, sent_i):
    """FInds adjective modifiers."""
    i = 0
    perturbations = []
    tokens_parse = list(parse)
    while i < len(tokens_parse):
        tokens = ''
        token = tokens_parse[i]

        while token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN' and \
                token.dep_ == 'amod':
            tokens = tokens + ' ' + token.text
            i += 1
            if i >= len(tokens_parse):
                break
            token = tokens_parse[i]

        if tokens:
            new_instance = remove_text(instance, sent_i,
                                       'Mod-Adj-NOUN',
                                       removed_text=tokens)
            if new_instance:
                perturbations.append(new_instance)

        i += 1
    return perturbations


def noun_modifiers(instance, parse, sent_i):
    """Finds noun modifiers."""
    perturbations = []
    i = 0
    tokens_parse = list(parse)
    while i < len(tokens_parse):
        tokens = ''
        token = tokens_parse[i]
        while token.dep_ == 'compound' and token.head.pos_ == 'NOUN' and \
                token.pos_ == 'NOUN':
            tokens = tokens + ' ' + token.text
            i += 1
            if i >= len(tokens_parse):
                break
            token = tokens_parse[i]

        if tokens:
            new_instance = remove_text(instance, sent_i, 'Mod-NOUN-NOUN',
                                       removed_text=tokens)
            if new_instance:
                perturbations.append(new_instance)

        i += 1

    return perturbations


def date_modifiers(instance, sent, sent_i):
    """Finds date modifiers."""
    perturbations = []
    for date in itertools.chain(regex_date_v1.finditer(sent[1]),
                                regex_date_v2.finditer(sent[1])):
        remove_texts = [date['day'], date['month'] + ' ' + date['day'] + ' ,',
                        date['day'] + ' ' + date['month']]
        type_date = ['day', 'month-day', 'day-month']
        for k, rm_text in enumerate(remove_texts):
            if k > 0:
                if rm_text + ' ,' in sent[1]:
                    rm_text = rm_text + ' ,'
                if ', ' + rm_text in sent[1]:
                    rm_text = ', ' + rm_text

            new_instance = remove_text(instance, sent_i,
                                       f'Mod-Date-{type_date[k]}',
                                       removed_text=rm_text)
            if new_instance:
                perturbations.append(new_instance)

    return perturbations


def write_adv(IDS_LABELS, adversary, type_adv):
    for_mturk = {
        'tag': adversary['tag'],
        'id': adversary['id'],
        'ds_id': adversary['ds_id'],
        'type': type_adv,
        'claim': adversary['claim'],
        'label': IDS_LABELS[adversary['label']],
        'evidence': format_mturk_text(adversary['mturk_text']),
        'adv_pred': str(adversary['adv_pred']),
        'orig_pred': str(adversary['orig_pred']),
        'orig_pred_logit': str(adversary['orig_pred_logit']),
        'adv_pred_logit': str(adversary['adv_pred_logit']),
        'orig_gold_logit': str(adversary['orig_gold_logit']),
        'adv_gold_logit': str(adversary['adv_gold_logit'])
    }
    return for_mturk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="Number of labels", type=int, default=6)
    parser.add_argument("--dataset", help="Dataset name",
                        choices=['fever', 'hover', 'vitaminc'],
                        default='fever')
    parser.add_argument("--dataset_dir", help="Path to the datasets folder",
                        default='data/', type=str)
    parser.add_argument("--model_path",
                        help="List of paths where the models are saved",
                        type=str, nargs='+')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--max_len", help="Max sequence encoding length", type=int,
                        default=512)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    claims_exclude = set()
    for file in os.scandir(f'ann/{args.dataset}_sent'):
        with open(file) as out:
            csv_reader = csv.DictReader(out)
            for i, line in enumerate(csv_reader):
                claim = line['claim']
                claims_exclude.add(claim)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    train, dev, test = get_datasets(args.dataset_dir,
                                    args.dataset,
                                    args.labels)

    random.shuffle(train.dataset)
    train.dataset = [train.dataset[i] for i in range(len(train))]
    if args.dataset != 'hover':
        train.dataset = [t for t in train.dataset if t['claim'] not in claims_exclude]
    train.dataset = train.dataset[:5000]

    random.shuffle(dev.dataset)
    dev.dataset = [dev.dataset[i] for i in range(len(dev))]
    if args.dataset != 'hover':
        dev.dataset = [t for t in dev.dataset if t['claim'] not in claims_exclude]
    dev.dataset = dev.dataset

    random.shuffle(test.dataset)
    test.dataset = [test.dataset[i] for i in range(len(test))]
    if args.dataset != 'hover':
        test.dataset = [t for t in test.dataset if t['claim'] not in claims_exclude]
    test.dataset = test.dataset

    models, tokenizers, configs, collate_fns = [], [], [], []
    for path in args.model_path:
        if '_bert_' in path:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.num_labels = args.labels

            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                config=config).to(device)

        elif '_roberta_' in path:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            config = RobertaConfig.from_pretrained('roberta-base')
            config.num_labels = args.labels

            model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                config=config).to(device)
        elif '_albert_' in path:
            tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
            config = AlbertConfig.from_pretrained('albert-base-v2')
            config.num_labels = args.labels

            model = AlbertForSequenceClassification.from_pretrained(
                'albert-base-v2',
                config=config).to(device)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        collate_fn = partial(collate_explanations,
                             tokenizer=tokenizer,
                             device=device,
                             pad_to_max_length=True,
                             max_length=args.max_len)

        models.append(model)
        tokenizers.append(tokenizer)
        configs.append(config)
        collate_fns.append(collate_fn)
    get_perturbations(args, models, tokenizers, collate_fns, test, 'test')
