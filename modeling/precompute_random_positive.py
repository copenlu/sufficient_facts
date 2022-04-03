import argparse
import random
import numpy as np
import torch
from modeling.data_reader import get_datasets, get_title
import json
from tqdm import tqdm
import copy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
stop_words = set(stopwords.words('english'))

def get_random(args, dataset):
    new_dataset = []
    for i, instance in enumerate(tqdm(dataset)):
        for _ in range(1):
            rand_idx = None
            claim_different = False
            while not rand_idx or rand_idx == i and not claim_different:
                rand_idx = random.randint(0, len(dataset)-1)
                if dataset[rand_idx]['claim'] != instance['claim']:
                    if all(sent in dataset[rand_idx]['text'] for sent in instance['text']):
                        continue
                    else:
                        claim_different = True

            new_instance = copy.deepcopy(instance)
            new_instance['rand_evidence'] = dataset[rand_idx]['text']
            new_dataset.append(new_instance)
    return new_dataset


def get_closest(dataset, instance, claims_tokens, i):
    max_overlap, closest_claim = -1, None
    for j, instance_ in enumerate(dataset):
        if i != j and instance_['claim'] != instance['claim'] and instance_['text'] != instance['text']:
            if all(sent in dataset[j]['text'] for sent in instance['text']):
                continue
            overlap = len(claims_tokens[i].intersection(claims_tokens[j]))
            if overlap > max_overlap:
                max_overlap = overlap
                closest_claim = j

    return closest_claim

from multiprocessing import Pool

def get_hard_negative(args, dataset):
    new_dataset = []
    p = Pool()
    claims_tokens = []
    for i, instance in enumerate(tqdm(dataset)):
        tokens = word_tokenize(instance['claim'])
        tokens = set([t for t in tokens if t not in stop_words and t not in punctuation])
        claims_tokens.append(tokens)

    closest_claims = p.starmap(get_closest, [(dataset, instance, claims_tokens, i) for i, instance in enumerate(dataset)])
    for i, instance in enumerate(tqdm(dataset)):
        new_instance = copy.deepcopy(instance)
        new_instance['close_claim_evidence'] = dataset[closest_claims[i]]['text']
        new_dataset.append(new_instance)
    return new_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['fever', 'hover', 'vitaminc'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/', type=str)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    train, dev, test = get_datasets(args.dataset_dir,
                                 args.dataset,
                                 args.labels)

    new_ds = get_random(args, train)
    new_ds = get_hard_negative(args, new_ds)

    with open(f'data/random_positive_train_{args.dataset}.json', 'w') as out:
        for line in new_ds:
            out.write(json.dumps(line)+'\n')

"""
python modeling/precompute_random_positive.py --dataset hover --dataset_dir data/ --labels 2

python3.8 modeling/precompute_positive.py --gpu --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --batch_size 8 --mode word
python3.8 modeling/precompute_positive.py --gpu --dataset hover --dataset_dir data/ --batch_size 8 --mode word
python3.8 modeling/precompute_positive.py --gpu --dataset vitaminc --dataset_dir data/vitaminc/ --batch_size 8 --mode word
"""