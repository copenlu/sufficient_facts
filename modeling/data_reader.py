"""Dataset objects and collate functions for all models and datasets."""
import json
from typing import Dict, List, Set, AnyStr
import bz2
import torch
import re
import os
import random
import unicodedata
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import spacy


LABEL_IDS_LIAR = {
    'pants-fire': 2, 'barely-true': 3, 'half-true': 4, 'mostly-true': 5,
    'false': 0, 'true': 1
}
THREE_LABEL_IDS_LIAR = {
    'pants-fire': 0, 'barely-true': 1, 'mostly-true': 1, 'false': 0, 'true': 2,
    'half-true': 1
}
FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
HOVER_LABELS = {'NOT_SUPPORTED': 1, 'SUPPORTED': 0}
LABEL_IDS_PUBH = {'false': 0, 'mixture': 1, 'true': 2, 'unproven': 3}


class FilteredDataset(Dataset):
    def filter_instances(self, split_ids_path):
        with open(split_ids_path) as out:
            ids_set = set([str(id_) for id_ in json.load(out)])
        print(len(ids_set))
        self.dataset = [i for i in self.dataset if str(i['id']) in ids_set]

    def __getitem__(self, item: int):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def get_title(text):
    return text.replace('-LRB-', '(').replace('-RRB-', ')').replace('_', ' ')


class FEVERDocDataset(FilteredDataset):
    def __init__(self, data_dir: str, wiki_dir: str):
        super().__init__()
        self.dataset = []

        if not os.path.exists(data_dir+'_proc'):
            print('Reading dataset instances...')

            self.wiki_docs = {}  # wiki_id: [wiki lines]

            for file in tqdm(os.scandir(wiki_dir),
                             desc='Reading wiki pages...', leave=True):
                # {"id": "", "text": "", "lines": ""}
                # "lines": "0\tThe following are the football -LRB-
                # soccer -RRB- events of the year 1928 throughout the world .\n1\t"
                with open(file) as out:
                    for doc in out:
                        doc = json.loads(doc)
                        doc_lines = []
                        for doc_line in doc['lines'].split('\n'):
                            cols = doc_line.split('\t')
                            if len(cols) > 1:
                                try:
                                    doc_lines.append([int(cols[0]), cols[1]])
                                except Exception as e:
                                    # print(cols)
                                    pass
                        self.wiki_docs[doc['id']] = doc_lines

            with open(data_dir) as out:
                for line in out:
                    line = json.loads(line)
                    if not line['claim'] or not line['evidence']:
                        continue
                    _dict = {}
                    _dict['id'] = line['id']
                    _dict['claim'] = line['claim']
                    _dict['label'] = FEVER_LABELS[line['label']]

                    _dict['text'] = []
                    for _s in line['evidence']:
                        wiki_url_text = unicodedata.normalize('NFC', _s[0])
                        _dict['text'] += [(get_title(_s[0]), [l[1] for l in self.wiki_docs[wiki_url_text]])]

                    self.dataset.append(_dict)

            with open(data_dir+'_proc', 'w') as out:
                out.write(json.dumps(self.dataset))

        else:
            self.dataset = json.load(open(data_dir + '_proc'))


class FeverDataset(FilteredDataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.dataset = []
        self.contr_ids = set()
        with open(data_dir) as out:
            for i, line in enumerate(out):
                line = json.loads(line)
                if not line['claim'] or not line['evidence']:
                    continue

                _dict = {}
                _dict['ds_id'] = line['id']
                _dict['id'] = i
                _dict['claim'] = line['claim']
                _dict['label'] = FEVER_LABELS[line['label']]
                _dict['title'] = get_title(line['evidence'][0][0])
                _dict['text'] = [(get_title(_s[0]), _s[2]) for _s in line['evidence']]

                self.dataset.append(_dict)

    def add_nei(self, file_path, contrastive=False):
        columns = ['id', 'gold', 'adv', 'gold_conf', 'adv_conf', 'claim', 'orig', 'adv', 'prediction_ok']
        df = pd.read_csv(file_path, sep='\t')
        df.columns = columns
        self.contr_ids = set()
        id2idx = {str(self.dataset[i]['id']): i for i in range(len(self.dataset))}
        for i, row in df.iterrows():
            if not contrastive:
                _dict = {}
                _dict['id'] = row['id']
                _dict['claim'] = row['claim']
                _dict['label'] = row['gold'] if row['prediction_ok'] == 'ok' else 2
                _dict['text'] = [row['adv']]
                self.dataset.append(_dict)
                self.contr_ids.add(str(row['id']))
            else:
                idx = id2idx[str(row['id'])]
                label = 1 if row['prediction_ok'] == 'ok' else -1
                # if row['prediction_ok'] == 'not':
                self.dataset[idx]['label_adv'] = label
                self.dataset[idx]['text_adv'] = row['adv']
                self.contr_ids.add(str(row['id']))

    def sample(self, n=10000):
        adv = [instance for i, instance in enumerate(self.dataset) if str(self.dataset[i]['id']) in self.contr_ids]
        rest = [instance for i, instance in enumerate(self.dataset) if str(self.dataset[i]['id']) not in self.contr_ids]
        rest = random.sample(rest, n)
        self.dataset = adv + rest

import pysbd
from string import punctuation

def strip(text):
    # text = text.replace('=', '\=')
    # text = mwparserfromhell.parse(text).strip_code()
    # text = text.replace('\=', '=')
    # text = parse_sentences(text)

    text = text.replace(' .', '.')
    return text


custom_punct = punctuation + ' ``''?.'


def get_sentences(text, seg):
    sentences = [s.strip() for s in seg.segment(text) if len(s.strip())]
    if len(sentences) == 1:
        return sentences
    final_sentences = []

    i = 0
    while i < len(sentences):
        only_punt = all(c in custom_punct for c in sentences[i])
        few_tokens = len(sentences[i].split()) < 3
        starts_lower = sentences[i+1].split()[0].islower() if i+1<len(sentences) else False
        if only_punt or few_tokens or starts_lower:
            if i == 0 or len(final_sentences) == 0:
                sentences[i+1] = sentences[i] + sentences[i+1]
            else:
                final_sentences[-1] = final_sentences[-1] + sentences[i]
        else:
            final_sentences.append(sentences[i])

        i += 1

    return final_sentences


class VitaminCDataset(FilteredDataset):

    def __init__(self, data_dir: str, sep_sentences=True):
        super().__init__()
        self.dataset = []

        # nlp = English()
        # sentencizer = nlp.create_pipe("sentencizer")
        # seg = pysbd.Segmenter(language="en", clean=False)

        with open(data_dir) as out:
            for i, line in tqdm(enumerate(out)):
                line = json.loads(line)
                if not line['claim'] or not line['evidence']:
                    continue
                # if 'test' in data_dir and 'revision_type' in line and line['revision_type'] != 'real':
                #     continue

                _dict = {}
                _dict['id'] = i
                _dict['ds_id'] = line['case_id']
                _dict['claim'] = line['claim']
                _dict['label'] = FEVER_LABELS[line['label']]
                _dict['text'] = []
                # text = strip(line['evidence'])
                # sentences = get_sentences(text, seg)
                # if len(sentences) > 1:
                #     print(line['claim'])
                #     print(sentences)

                # _dict['text'] = []
                # for j, sent in enumerate(sentences):
                #     title = line['page'] if j == 0 else ''
                #     _dict['text'] += [(title, sent)]

                _dict['text']= [(line['page'], line['evidence'])]

                self.dataset.append(_dict)

ann_mapping = {'ENOUGH -- IRRELEVANT': 0, 'ENOUGH -- REPEATED': 0, 'NOT ENOUGH': 1}

LABEL_IDS = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2, 'NOT_SUPPORTED': 1, 'SUPPORTED': 0}


class JsonlDataset(FilteredDataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.dataset = []

        with open(data_dir) as out:
            for i, line in enumerate(out):
                if not line.strip():
                    continue
                line = json.loads(line)
                self.dataset.append(line)

class NEIDataset(FilteredDataset):
    def __init__(self, data_dir: str, nei_label_id=2):
        super().__init__()
        self.dataset = []

        with open(data_dir) as out:
            for i, line in enumerate(out):
                if not line.strip():
                    continue
                line = json.loads(line)
                _dict = {}
                _dict['id'] = i
                # _dict['ds_id'] = line['id']
                _dict['claim'] = line['claim']
                _dict['label'] = LABEL_IDS[line['label_before']] if line['label_after'] != 'NOT ENOUGH' else nei_label_id
                _dict['text'] = line['evidence']
                _dict['nei_ann'] = line['agreement']
                _dict['type'] = line['type']
                _dict['removed'] = line['removed']
                _dict['text_orig'] = line['text_orig']
                _dict['label_before'] = line['label_before']
                _dict['label_after'] = line['label_after']

                self.dataset.append(_dict)


def read_wiki_doc(doc_path, line_num, sentence_num):
    with bz2.open(doc_path) as out:
        for i, line in enumerate(out):
            if i == line_num:
                doc = json.loads(line)
                try:
                    return sum(doc['text'], [])[1:][sentence_num], doc['title']
                except:
                    return '', doc['title']


class HoverDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.dataset = []

        if not os.path.exists(data_dir+'_proc'):
            print('Reading doc mapping...')
            with open('data/doc_mapping.json') as out:
                doc_location = json.load(out)

            print('Reading dataset instances...')
            with open(data_dir) as out:
                json_doc = json.load(out)

                for i, line in tqdm(enumerate(json_doc)):
                    _dict = {}
                    _dict['ds_id'] = line['uid']
                    _dict['id'] = i
                    _dict['claim'] = line['claim']
                    _dict['label'] = HOVER_LABELS[line["label"]]
                    _dict['text'] = []
                    for fact in line['supporting_facts']:
                        loc = doc_location[fact[0]]
                        s, title = read_wiki_doc(loc[0], loc[1], fact[1])
                        s = re.sub(r'\<a href\=.*?\>', '[[', s)
                        s = re.sub(r'\<\/a\>', ']]', s)
                        _dict['text'].append((title, s))
                    self.dataset.append(_dict)

            with open(data_dir+'_proc', 'w') as out:
                out.write(json.dumps(self.dataset))
        else:
            self.dataset = json.load(open(data_dir+'_proc'))
            # for i in range(len(self.dataset)):
            #     self.dataset[i]['ds_id'] = deepcopy(self.dataset[i]['id'])
            #     self.dataset[i]['id'] = i


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class ArrayDataset(Dataset):
    def __init__(self, array):
        super().__init__()
        self.dataset = array

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def get_datasets(dataset_dir: str,
                 dataset_type: str,
                 num_labels: int = None):
    datasets = []

    if dataset_type == 'fever':
        dataset_list = ['train_nli.jsonl',
                        'dev_nli.jsonl',
                        'test_nli.jsonl']
        for ds_name in dataset_list:
            ds = FeverDataset(f'{dataset_dir}/{ds_name}')
            datasets.append(ds)
    elif dataset_type == 'vitaminc':
        dataset_list = ['train.jsonl',
                        'dev.jsonl',
                        'test_real.jsonl']
        for ds_name in dataset_list:
            ds = VitaminCDataset(f'{dataset_dir}/{ds_name}')
            datasets.append(ds)

    elif dataset_type == 'fever_doc':
        dataset_list = ['train_nli.jsonl',
                        'dev_nli.jsonl',
                        'test_nli.jsonl']
        for ds_name in dataset_list:
            ds = FEVERDocDataset(f'{dataset_dir}/{ds_name}',
                                 '../fever-adversarial-attacks/data/wiki-pages/wiki-pages/')
            datasets.append(ds)

    elif dataset_type == 'hover':
        #  train, test = train_test_split(train_json, test_size=0.1, random_state=1)
        dataset_list = ['hover_train.json',
                        'hover_dev.json',
                        'hover_dev_release_v1.1.json']
        for ds_name in dataset_list:
            ds = HoverDataset(f'{dataset_dir}/{ds_name}')
            datasets.append(ds)
    else:
        raise ValueError(f'No such dataset {dataset_type}')

    return tuple(datasets)


def collate_explanations(instances: List[Dict],
                         tokenizer: AutoTokenizer,
                         max_length: int, max_claim_len:int=None,
                         pad_to_max_length: bool = True, claim_only=False,
                         device='cuda'):
    """Collates a batch with data from an explanations dataset"""

    # [CLS] claim tokens [SEP] sentence1 tokens [SEP] sentence2 tokens ... [SEP]
    input_ids = []
    sentence_start_ids = []
    claim_ends = []
    titles = []
    for instance in instances:
        titles_instance = []
        instance_sentence_starts = []
        instance_input_ids = [tokenizer.cls_token_id]
        claim_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(instance['claim']))
        if max_claim_len:
            claim_tokens = claim_tokens[:max_claim_len]
        instance_input_ids.extend(claim_tokens)
        instance_input_ids.append(tokenizer.sep_token_id)

        claim_ends.append(len(instance_input_ids))
        if claim_only:
            input_ids.append(instance_input_ids)
            sentence_start_ids.append(instance_sentence_starts)
            titles.append(titles_instance)
        else:
            for i, sentence in enumerate(instance['text']):
                title = None
                if type(sentence) is tuple or isinstance(sentence, list):
                    sentence, title = sentence[:2]
                    titles_instance.append(title)

                instance_sentence_starts.append(len(instance_input_ids))

                if title:
                    sentence = title + ' ' + sentence
                sentence_tokens = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(sentence))
                instance_input_ids.extend(sentence_tokens)
                instance_input_ids.append(tokenizer.sep_token_id)

            input_ids.append(instance_input_ids)

            sentence_start_ids.append(instance_sentence_starts)
            titles.append(titles_instance)

    if pad_to_max_length:
        batch_max_len = max_length
    else:
        batch_max_len = max([len(_s) for _s in input_ids])

    input_ids = [_s[:batch_max_len] for _s in input_ids]
    sentence_start_ids = [[i for i in ids if i < batch_max_len]
                          for ids in sentence_start_ids]

    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (
                batch_max_len - len(_s)) for _s in
         input_ids])

    labels = torch.tensor([_x['label'] for _x in instances],
                          dtype=torch.long)

    result = {
        'titles': titles,
        'input_ids_tensor': padded_ids_tensor.cuda(device),
        'target_labels_tensor': labels.cuda(device),
        'sentence_start_ids': sentence_start_ids,
        'ids': [instance['id'] for instance in instances],
        'query_claim_ends': torch.tensor(claim_ends).cuda(device)
    }

    return result



