import argparse
import json
from typing import Dict
from modeling.data_reader import get_datasets
from copy import deepcopy
import benepar, spacy
from tqdm import tqdm
from modeling import constituent_removal


def remove_text_custom(instance, sentence_i, tag, removed_text, new_text=None):
    num_occurrences = sum(instance[f'all_sents_{args.mode}'][sentence_i][1][
                          i:i + len(removed_text)] == removed_text for i in
                          range(len(instance[f'all_sents_{args.mode}'][sentence_i][1])))

    if num_occurrences != 1:
        return None
    if removed_text.strip().lower().startswith('of '):
        return None

    removed_text = removed_text.strip()

    new_sent = instance[f'all_sents_{args.mode}'][sentence_i][1].replace(removed_text, ' ').replace('  ', ' ').replace('  ', ' ') if removed_text else new_text

    return {'new_sent': new_sent, 'type': tag, 'removed': removed_text}


def get_perturbations(args, dataset, split) -> (Dict, Dict):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    if args.mode == 'word':
        positives = json.load(open(f'data/pos_precomp_train_{args.dataset}_word.json'))
    elif args.mode == 'vector':
        positives = json.load(open(f'data/pos_precomp_train_{args.dataset}_vector.json'))

    for instance_i, instance in enumerate(tqdm(dataset)):
        instance['orig_text'] = deepcopy(instance['text'])
        if args.mode == 'orig':
            all_sents = [(text[0], text[1], 'orig', 1) for text in instance['orig_text']]
        else:
            instance_positives = positives[instance['claim']][f'positive_sent_{args.mode}']
            instance[f'pos_sents_{args.mode}'] = instance_positives
            all_sents = [(text[0], text[1], f'pos_{args.mode}', text[2]) for text in instance_positives]

        instance[f'all_sents_{args.mode}'] = all_sents

        instance[f'{args.mode}_sent_perturbs'] = {}
        for sent_i, sent in enumerate(all_sents):
            sent_1 = sent[1].replace('-LRB- ', '(').replace('-LSB- ', '[').replace(' -RSB-', ']').replace(' -RRB-', ')')
            used_pps = []
            sent_perturbations = []

            try:
                doc = nlp(sent_1)
            except Exception as e:
                print(e)
                continue
            parse = list(doc.sents)
            if not parse:
                continue
            parse = parse[0]

            sent_perturbations += constituent_removal.date_modifiers(instance, sent, sent_i)
            sent_perturbations += constituent_removal.noun_modifiers(instance, parse, sent_i)
            sent_perturbations += constituent_removal.adjective_modifiers(instance, parse, sent_i)
            sent_perturbations += constituent_removal.adverb_modifiers(instance, parse, sent_i)

            queue = [(parse, False)]
            while queue:
                item, pp_used = queue.pop(0)

                tags = [ch[0].tag_ for ch in item._.children if
                                len(ch._.labels) == 0]

                all_children = list(item._.children)

                if len(item._.labels) > 0:
                    pp_perturb = constituent_removal.pp_adjunct(all_children, instance, sent_i)
                    pp_perturb_fitered = []
                    for pp in pp_perturb:
                        if any(pp['removed'] in used_pp for used_pp in used_pps):
                            continue
                        else:
                            pp_perturb_fitered.append(pp)
                            used_pps.append(pp['removed'])

                    sent_perturbations += pp_perturb_fitered

                    sent_perturbations += constituent_removal.number_modifiers(all_children, instance, sent_i, tags, item)
                    sent_perturbations += constituent_removal.sbar(all_children, instance, item, sent, sent_i)

                    for const in item._.children:
                        queue.append((const, pp_used))

            instance[f'{args.mode}_sent_perturbs'][sent_i] = sent_perturbations

    with open(f'data/{args.dataset}_{split}_{args.mode}_precomputed_perturbations.json', 'w') as out:
        json.dump(dataset.dataset, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['fever', 'hover', 'vitaminc'],
                        default='liar')
    parser.add_argument("--mode", help="Which sentences to compute perturbations for",
                        choices=['orig', 'word', 'vector'],
                        default='orig')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/', type=str)
    parser.add_argument("--max_len", help="Learning Rate", type=int,
                        default=512)

    args = parser.parse_args()
    print(args)
    constituent_removal.remove_text = remove_text_custom
    train, _, _ = get_datasets(args.dataset_dir,
                                 args.dataset,
                                 args.labels)

    get_perturbations(args, train, 'train')

"""
python3.8 modeling/precompute_perturbations.py --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --labels 3 --mode orig
python3.8 modeling/precompute_perturbations.py --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --labels 3 --mode word
python3.8 modeling/precompute_perturbations.py --dataset hover --dataset_dir data/ --labels 2 --mode orig
python3.8 modeling/precompute_perturbations.py --dataset hover --dataset_dir data/ --labels 2 --mode word
python3.8 modeling/precompute_perturbations.py --dataset vitaminc --dataset_dir data/vitaminc/ --labels 3 --mode orig
python3.8 modeling/precompute_perturbations.py --dataset vitaminc --dataset_dir data/vitaminc/ --labels 3 --mode word
"""