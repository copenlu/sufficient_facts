"""
Precompute predictions of models for original - removed constituent or sentence.
"""
import argparse
import random
from functools import partial
import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizerFast, RobertaForSequenceClassification, RobertaConfig, RobertaTokenizerFast, AlbertForSequenceClassification, AlbertConfig, AlbertTokenizerFast
from modeling.data_reader import collate_explanations, get_datasets
from tqdm import tqdm
import json
from copy import deepcopy

FEVER_IDS = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
HOVER_IDS = {1: 'NOT_SUPPORTED', 0: 'SUPPORTS'}


def get_predictions(model, tokenizer, collate_fn, dataset):
    NEI_LABEL = 1 if args.dataset == 'hover' else 2


    softmax = torch.nn.Softmax(dim=1)
    new_ds = []
    for instance in tqdm(dataset):
        # if instance['label'] == NEI_LABEL:
        #     continue
        perturbs = []
        for i, sent in enumerate(instance['text']):
            if len(instance['text']) > 1:
                new_instance = deepcopy(instance)
                new_instance['orig_text'] = deepcopy(new_instance['text'])
                new_instance['text'] = new_instance['text'][:i] + new_instance['text'][i+1:]
                perturbs.append(new_instance)

        if not perturbs:
            continue

        preds, logits, logit_pred = [], [], []
        for k in range(0, len(perturbs), args.batch_size):
            perturbs_batch = perturbs[k:k+args.batch_size]
            batch = collate_fn(perturbs_batch)
            logits_instance = model(batch['input_ids_tensor'],
                   attention_mask=batch[
                                      'input_ids_tensor'] !=
                                  tokenizer.pad_token_id).logits
            logits_instance = softmax(logits_instance).detach()
            preds += torch.argmax(logits_instance, dim=-1).cpu().numpy().tolist()
            logits += logits_instance.cpu().numpy().tolist()
            logit_pred += [instance_logits[preds[i]] for i, instance_logits in
              enumerate(logits)]

        instance['sent_perturbs'] = {}
        for i, sent in enumerate(instance['text']):
            instance['sent_perturbs'][str(i)] = {}
            instance['sent_perturbs'][str(i)][f'pred_{args.model_name}'] = preds[i]
            instance['sent_perturbs'][str(i)][f'logits_{args.model_name}'] = logits[i]
            instance['sent_perturbs'][str(i)][f'logit_pred_{args.model_name}'] = logit_pred[i]
        new_ds.append(instance)

    return new_ds


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
    parser.add_argument("--model_path",
                        help="Path where the model is serialized",
                        type=str)
    parser.add_argument("--model_name",
                        help="Name of the model",
                        type=str)
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
    # with open(f'data/{args.dataset}_train_orig_precomputed_perturbations.json') as out:
    #     train.dataset = json.load(out)

    if '_bert_' in args.model_path:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = args.labels

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              config=config).to(device)

    elif '_roberta_' in args.model_path:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = args.labels

        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            config=config).to(device)
    elif '_albert_' in args.model_path:
        tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        config = AlbertConfig.from_pretrained('albert-base-v2')
        config.num_labels = args.labels

        model = AlbertForSequenceClassification.from_pretrained(
            'albert-base-v2',
            config=config).to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    collate_fn = partial(collate_explanations,
                         tokenizer=tokenizer,
                         device=device,
                         pad_to_max_length=True,
                         max_length=args.max_len)

    new_ds = get_predictions(model, tokenizer, collate_fn, train)
    with open(f'data/{args.dataset}_{args.model_name}_train_precomputed_sentence_predictions.json', 'w') as out:
        json.dump(new_ds, out)



"""
python3.8 modeling/precompute_sentence_predictions.py --gpu --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --model_path fever_oracle_bert_1e5_1 --labels 3 --batch_size 8 --model_name bert_1
python3.8 modeling/precompute_predictions.py --gpu --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --model_path models/fever_oracle_roberta_1e5 --labels 3 --batch_size 8 --model_name roberta
python3.8 modeling/precompute_predictions.py --gpu --dataset fever --dataset_dir /image/image-copenlu/generating-adversarial-claims/data/ --model_path models/fever_oracle_albert_1e5 --labels 3 --batch_size 8 --model_name albert

python3.8 modeling/precompute_predictions.py --gpu --dataset hover --dataset_dir data/ --model_path models/hover_bert_1e5 --labels 2 --batch_size 8 --model_name bert
python3.8 modeling/precompute_predictions.py --gpu --dataset hover --dataset_dir data/ --model_path models/hover_roberta_1e5 --labels 2 --batch_size 8 --model_name roberta
python3.8 modeling/precompute_predictions.py --gpu --dataset hover --dataset_dir data/ --model_path models/hover_albert_1e5 --labels 2 --batch_size 8 --model_name albert

python3.8 modeling/precompute_predictions.py --gpu --dataset vitaminc --dataset_dir data/vitaminc/ --model_path models/vitaminc_bert_1e5 --labels 3 --batch_size 8 --model_name bert
python3.8 modeling/precompute_predictions.py --gpu --dataset vitaminc --dataset_dir data/vitaminc/ --model_path models/vitaminc_roberta_1e5 --labels 3 --batch_size 8 --model_name roberta
python3.8 modeling/precompute_predictions.py --gpu --dataset vitaminc --dataset_dir data/vitaminc/ --model_path models/vitaminc_albert_1e5 --labels 3 --batch_size 8 --model_name albert
"""