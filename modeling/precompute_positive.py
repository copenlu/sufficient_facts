import argparse
import random
import numpy as np
import torch
from typing import List
from string import punctuation
from transformers import AutoTokenizer, AutoModel
import heapq
from modeling.data_reader import get_datasets, get_title
import re
import os
import json
from tqdm import tqdm
import bz2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize


stop_words = set(stopwords.words('english'))
re_start = re.compile(r'\<a href\=.*?\>')


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_tokens_set(tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = set([s for s in tokens if s not in punctuation])
    return tokens


def get_doc_sentences(doc_ids: List[str], current_sent: List[str], doc_location):
    res = []

    for doc_id in doc_ids:
        title = doc_id
        if title not in doc_location:
            continue
        sents = doc_location[title]
        sents = [s.strip() for s in sents if len(s) > 3]
        sents = [re_start.sub('', s) for s in sents]
        sents = [s.replace('</a>', '') for s in sents]
        sents = [s.replace(u'\xa0', u' ') for s in sents if len(s) > 3]
        sents = [s for s in sents if not s[0] in punctuation]

        res += [(title, s) for s in sents]
    return res


def get_sim(claim_tokens, sent):
    sent_tokens = word_tokenize(sent[1])
    sent_tokens = set([t.lower() for t in sent_tokens if t.lower() not in stop_words])
    if sent_tokens == 0:
        return -1
    sim = len(claim_tokens.intersection(sent_tokens))
    return sim


def get_closest_sents_bow(claim: str, docs_sentences: List[str]):
    claim_tokens = set([t.lower() for t in word_tokenize(claim) if t.lower() not in stop_words])
    results = [get_sim(claim_tokens, doc) for doc in docs_sentences]
    results = [(sent[0], sent[1], sim) for sent, sim in zip(docs_sentences, results)]

    return heapq.nlargest(3, results, key=lambda x: x[-1]) if results else []


def get_closest_sents(claim: str, docs_sentences: List[str], model, tokenizer):
    encoded_claim = tokenizer(claim, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output_claim = model(**encoded_claim)
    sentence_embeddings_claim = mean_pooling(model_output_claim, encoded_claim['attention_mask'])
    results = []
    docs_sentences = [sent for sent in docs_sentences if len(sent[1].strip())>1]
    for i in range(0, len(docs_sentences), args.batch_size):
        sentences_batch = docs_sentences[i:i+args.batch_size]
        encoded_sentences = tokenizer([ds[1] for ds in sentences_batch],
                                      padding=True,
                                      truncation=True,
                                      max_length=150,
                                      return_tensors='pt').to(device)
        with torch.no_grad():
            model_output_sentences = model(**encoded_sentences)
        sentence_embeddings_sents = mean_pooling(model_output_sentences, encoded_sentences['attention_mask'])
        sim = torch.cosine_similarity(sentence_embeddings_claim, sentence_embeddings_sents, dim=1)

        for j in range(len(sentences_batch)):
            results.append((docs_sentences[j][0], docs_sentences[j][1], sim[j].item()))

    return heapq.nlargest(3, results, key=lambda x: x[-1]) if results else []


def get_positives(args, model, tokenizer, dataset, doc_location):
    for instance in tqdm(dataset):
        doc_ids = list(set([instance['text'][i][0] for i in range(len(instance['text']))]))
        current_sents = [instance['text'][i][1] for i in range(len(instance['text']))]
        doc_sentences = get_doc_sentences(doc_ids, current_sents, doc_location)
        if args.mode == 'word':
            closest_sents = get_closest_sents_bow(instance['claim'], doc_sentences)
        else:
            closest_sents = get_closest_sents(instance['claim'], doc_sentences, model, tokenizer)

        current_tokens_sets = [get_tokens_set(tokenizer, s) for s in current_sents]
        sent_tokens_set = [get_tokens_set(tokenizer, s) for _, s, _ in closest_sents]
        closest_sents = [s for s, s_set in zip(closest_sents, sent_tokens_set)
                 if all(len(s_set.intersection(current_s)) / len(s_set) < 0.9 for current_s in current_tokens_sets)]

        instance[f'positive_sent_{args.mode}'] = closest_sents

    return {instance['claim']: instance for instance in dataset}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['fever', 'hover', 'vitaminc'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/', type=str)
    parser.add_argument("--mode", choices=['word', 'vector'])
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        type=str, nargs='+')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--max_len", help="Learning Rate", type=int,
                        default=512)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    train, dev, test = get_datasets(args.dataset_dir,
                                 args.dataset,
                                 args.labels)

    wiki_docs_path = f'data/{args.dataset}_wiki_docs_mapping.json'
    if os.path.exists(wiki_docs_path):
        print('Loading serialized documents... ', flush=True)
        wiki_docs = json.load(open(wiki_docs_path))
    else:
        wiki_docs = {}  # wiki_id: [wiki lines]
        if args.dataset == 'fever':
            for file in tqdm(os.scandir('wiki-pages'), desc='Reading wiki pages...', leave=True):
                with open(file) as out:
                    for doc in out:
                        doc = json.loads(doc)
                        doc_lines = []
                        for doc_line in doc['lines'].split('\n'):
                            cols = doc_line.split('\t')
                            if len(cols) > 1:
                                try:
                                    doc_lines.append(cols[1])
                                except Exception as e:
                                    pass
                        wiki_docs[get_title(doc['id'])] = doc_lines
        elif args.dataset == 'hover':
            with open('data/doc_mapping.json') as out:
                doc_locations = json.load(out)
            wiki_docs = {}

            def get_instance_doc(title, doc_path, line_num):
                with bz2.open(doc_path) as out:
                    for i, line in enumerate(out):
                        if i == line_num:
                            doc = json.loads(line)
                            try:
                                return title, sum(doc['text'], [])[1:]
                            except:
                                print('Empty', flush=True)
                                return title, ['']

            wiki_titles = set()
            for instance in train:
                for sent in instance['text']:
                    if sent[0] not in wiki_docs:
                        wiki_titles.add(sent[0])

            print(f'Extracting page contents of {len(wiki_titles)} pages...', flush=True)
            with Pool() as p:
                wiki_docs_list = p.starmap(get_instance_doc, [(title, doc_locations[title][0],
                                                           doc_locations[title][1]) for title in wiki_titles])
            wiki_docs = {k: v for k, v in wiki_docs_list}

        elif args.dataset == 'vitaminc':
            wiki_docs = {}
            if os.path.exists('data/fever_wiki_docs_mapping.json'):
                wiki_docs.update(json.load(open('data/fever_wiki_docs_mapping.json')))
            if os.path.exists('data/hover_wiki_docs_mapping.json'):
                wiki_docs.update(json.load(open('data/hover_wiki_docs_mapping.json')))

            def get_page(title):
                import wikipedia
                try:
                    return title, wikipedia.page(title).content
                except:
                    return title, ''

            titles = list(set([sent[0] for instance in train for sent in instance]))
            title = [title for title in titles if title not in wiki_docs]
            with Pool() as p:
                docs = p.map(get_page, titles)
                docs_sentences = p.map(sent_tokenize, [d[1] for d in docs])
            wiki_docs.update({k[0]: v for k, v in zip(docs, docs_sentences)})

        with open(wiki_docs_path, 'w') as out:
            json.dump(wiki_docs, out)

    if args.mode == 'vector':
        model = AutoModel.from_pretrained("sentence-transformers/roberta-large-nli-stsb-mean-tokens").to(device)
    else:
        model = None

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/roberta-large-nli-stsb-mean-tokens")
    pos_train = get_positives(args, model, tokenizer, train, wiki_docs)
    json.dump(pos_train, open(f'data/pos_precomp_train_{args.dataset}_{args.mode}.json', 'w'))

"""
python3.8 modeling/precompute_positive.py --gpu --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --batch_size 8 --mode word
python3.8 modeling/precompute_positive.py --gpu --dataset hover --dataset_dir data/ --batch_size 8 --mode word
python3.8 modeling/precompute_positive.py --gpu --dataset vitaminc --dataset_dir data/vitaminc/ --batch_size 8 --mode word
"""