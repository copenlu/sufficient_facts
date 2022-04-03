import argparse
import math
import random
from argparse import Namespace
from functools import partial
from typing import Dict
import numpy as np
import torch
import copy
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, \
    RobertaForSequenceClassification, RobertaTokenizerFast, RobertaConfig,\
    AlbertForSequenceClassification, AlbertTokenizerFast, AlbertConfig
from transformers.optimization import AdamW
from modeling.data_reader import collate_explanations, get_datasets, FilteredDataset, NEIDataset
import json
from collections import Counter, defaultdict

def train_model(args: Namespace,
                model: torch.nn.Module,
                train_dl: DataLoader, dev_dl: DataLoader,
                optimizer: torch.optim.Optimizer, scheduler) -> (Dict, Dict):

    best_score, best_model_weights = {'dev_target_f1': 0, 'loss_dev': 1}, None
    loss_fct = torch.nn.CrossEntropyLoss()
    model.train()
    for ep in range(args.epochs):
        step = -1
        for batch_i, batch in enumerate(train_dl):
            step += 1
            logits = model(batch['input_ids_tensor'], attention_mask=batch['input_ids_tensor']!=tokenizer.pad_token_id).logits

            loss = loss_fct(logits.view(-1, args.labels),
                            batch['target_labels_tensor'].long().view(-1)) / args.accum_steps
            loss.backward()

            if (batch_i + 1) % args.accum_steps == 0:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

            current_train = {
                'train_loss': loss.cpu().data.numpy(),
                'epoch': ep,
                'step': step,
            }
            print(
                '\t'.join([f'{k}: {v:.3f}' for k, v in current_train.items()]), flush=True, end='\r')

            if step > 0 and step == ((step_per_epoch)//2):
                current_val = eval_model(args, model, dev_dl)
                print(f"epoch {ep}, step {step}", current_val, flush=True)
                result = eval_model(args, model, test_dl)
                print('Test on test ds', result, flush=True)
                result = eval_model(args, model, test_dl_nei)
                print('Test on nei ds', result, flush=True)

                if current_val['dev_target_f1'] >= best_score['dev_target_f1']:
                    best_score = copy.deepcopy(current_val)
                    best_model_weights = copy.deepcopy(model.state_dict())
                model.train()

        current_val = eval_model(args, model, dev_dl)
        current_val.update(current_train)
        print(f"epoch {ep}, step {step}", current_val, flush=True)
        result = eval_model(args, model, test_dl)
        print('Test on test ds', result, flush=True)
        result = eval_model(args, model, test_dl_nei)
        print('Test on nei ds', result, flush=True)
        if current_val['dev_target_f1'] >= best_score['dev_target_f1']:
            best_score = copy.deepcopy(current_val)
            best_model_weights = copy.deepcopy(model.state_dict())

        model.train()
    return best_model_weights, best_score


def eval_target(args,
                model: torch.nn.Module,
                test_dl: DataLoader):
    model.eval()
    pred_class, true_class, losses, ids = [], [], [], []
    inputs = []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluation"):
            optimizer.zero_grad()
            logits = model(batch['input_ids_tensor'], attention_mask=batch['input_ids_tensor']!=tokenizer.pad_token_id).logits
            inputs += [tokenizer.decode(i) for i in batch['input_ids_tensor'].detach().cpu().numpy().tolist()]
            true_class += batch['target_labels_tensor'].detach().cpu().numpy().tolist()
            pred_class += logits.detach().cpu().numpy().tolist()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, args.labels),
                            batch['target_labels_tensor'].long().view(-1))
            ids += batch['ids']
            losses.append(loss.item())

        prediction_orig = np.argmax(np.asarray(pred_class).reshape(-1, args.labels),
                               axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(true_class,
                                                      prediction_orig,
                                                      average='macro')

        majority_label = Counter(true_class).most_common(1)[0][0]
        majority = [majority_label] * len(true_class)
        print('Majority', precision_recall_fscore_support(true_class, majority, average='macro'))

        acc = accuracy_score(true_class, prediction_orig)

    return prediction_orig, ids, np.mean(losses), acc, p, r, f1


def eval_model(args,
               model: torch.nn.Module,
               test_dl: DataLoader):
    prediction, ids, losses, acc, p, r, f1 = eval_target(args, model, test_dl)
    dev_eval = {
        'loss_dev': np.mean(losses),
        'dev_target_p': p,
        'dev_target_recall': r,
        'dev_target_f1': f1,
        'dev_acc': acc
    }

    return dev_eval


def add_precomputed(args, dataset, nei_id):
    positive_key = f'pos_sents_{args.positives_type}'
    # postive_perturb_key = f'{args.positives_type}_sent_perturbs'
    additional_instances = []
    ds_id_positive_sentences = {}
    print("Initial dataset size", len(dataset.dataset))
    additional_positives = defaultdict(lambda: [])
    additional_negatives = []
    additional_sent_negatives = defaultdict(lambda: [])
    additional_nei = defaultdict(lambda: [])
    # additional_n_const_pos = defaultdict(lambda: [])
    additional_n_const_pos = []
    with open(args.perturbations) as out:
        lines = json.load(out)
        for line in lines:
            # line = json.loads(line)
            if str(line['ds_id']) in additional_positives:
                continue

            claim_tokens = set([w.lower() for w in word_tokenize(line['claim'])])
            pos_overlap = []
            # perturb_overlap = []
            for pos_sent_id, pos_sent in enumerate(line['pos_sents_word']):
                pos_sent_overlap_num = len(set([w.lower() for w in word_tokenize(pos_sent[1])]).intersection(claim_tokens))
                pos_overlap.append((pos_sent, pos_sent_overlap_num))
                # for perturb_id, perturb in enumerate(line[postive_perturb_key].get(str(pos_sent_id), [])):
                #     perturb_overlap_num = len(set([w.lower() for w in word_tokenize(perturb['removed'])]).intersection(claim_tokens))
                #     perturb_sent = copy.deepcopy(pos_sent)
                #     perturb_sent[1] = perturb['new_sent']
                #     perturb_overlap.append((perturb_sent, perturb_overlap_num))  # TODO add min overlap as hyperparam

            pos_overlap = list(sorted(pos_overlap, key=lambda x: x[-1], reverse=True))[:args.max_positives]
            # perturb_overlap = list(sorted(perturb_overlap, key=lambda x: x[-1], reverse=True))[:args.max_positives]
            all_positives = []
            if 'p_sent' in args.positives_mode and pos_overlap:
                all_positives += pos_overlap
            # if 'p_sent_const' in args.positives_mode and perturb_overlap:
            #     all_positives += perturb_overlap
            if all_positives:
                ds_id_positive_sentences[line['ds_id']] = all_positives

            for new_sent, overlap in all_positives:
                instance_new = copy.deepcopy(line)
                rand_pos = random.randint(0, len(instance_new['text'])-1)
                instance_new['text'] = instance_new['text'][:rand_pos] + [new_sent] + instance_new['text'][rand_pos:]
                additional_positives[str(line['ds_id'])].append(instance_new)

            if 'n_sent_pos' in args.negatives_mode and pos_overlap:
                for s in pos_overlap[:args.max_negatives]:
                    negative = copy.deepcopy(line)
                    negative['text'] = [s[0]]
                    negative['label'] = nei_id
                    # negative['ds_id'] = line['ds_id']
                    negative['overlap'] = s[1]
                    # additional_n_const_pos[str(line['ds_id'])].append(negative)
                    # additional_n_const_pos[str(line['ds_id'])].append(negative)
                    additional_n_const_pos.append(negative)

    # collect all predictions on the negative augmentations
    if 'n_const' in args.negatives_mode or 'n_const_pos' in args.negatives_mode:
        ds_id_instance = {}
        dsid_sentid_newsent_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
        for i, (model_name, model_preds) in enumerate(zip(args.other_models, args.serialized_predictions_path)):
            with open(model_preds) as out:
                for instance in json.load(out):
                    ds_id = str(instance['ds_id'])
                    ds_id_instance[ds_id] = instance
                    for sent_id in instance['orig_sent_perturbs'].keys():
                        for sent in instance['orig_sent_perturbs'][sent_id]:
                            new_sent = sent['new_sent']
                            if f'logits_{model_name}' in sent:
                                pred = sent[f'logits_{model_name}']
                                dsid_sentid_newsent_predictions[ds_id][sent_id][new_sent] += [pred]

        for ds_id, sentences in dsid_sentid_newsent_predictions.items():  # TODO sort negatives by disagr/conf
            for sent_id, new_sentences in sentences.items():
                for new_sent, predictions in new_sentences.items():
                    instance = ds_id_instance[ds_id]
                    preds = [np.argmax(predictions[i]) for i in range(len(predictions))]

                    if instance['label'] == NEI_LABEL and args.use_nei:
                        if preds.count(NEI_LABEL) != 3:
                            instance_new = copy.deepcopy(instance)
                            instance_new['text'][int(sent_id)][1] = new_sent
                            instance_new['label'] = NEI_LABEL
                            instance_new['cl'] = 'pos'
                            additional_nei[ds_id].append(instance_new)

                    elif instance['label'] != NEI_LABEL:
                        predicted_majority, count = Counter(preds).most_common(1)[0]
                        preds_avg = [np.mean([predictions[i][j] for i in range(len(predictions))]) for j in range(len(predictions[0]))]
                        if count >= args.min_nei_counts and predicted_majority == NEI_LABEL and preds_avg[NEI_LABEL] >= args.min_nei_preds and int(sent_id) < len(instance['text']):
                            if 'n_const' in args.negatives_mode:
                                instance_new = copy.deepcopy(instance)
                                instance_new['text'][int(sent_id)][1] = new_sent
                                instance_new['label'] = NEI_LABEL
                                instance_new['cl'] = 'neg'
                                instance_new['pred_avg'] = preds_avg[NEI_LABEL]
                                instance_new['ds_id'] = ds_id
                                instance_new['pred_avg'] = preds_avg[NEI_LABEL]
                                additional_negatives.append(instance_new)

    new_sent_instances = []
    if 'n_sent' in args.negatives_mode:
        ds_id_instance = {}
        dsid_sentid_newsent_predictions_sents = defaultdict(lambda: defaultdict(lambda: []))
        for i, (model_name, model_preds) in enumerate(zip(args.other_models, args.serialized_sentence_predictions_path)):
            with open(model_preds) as out:
                for instance in json.load(out):
                    ds_id = str(instance['ds_id'])
                    ds_id_instance[ds_id] = instance
                    if 'sent_perturbs' not in instance:
                        continue
                    for sent_id in instance['sent_perturbs'].keys():
                        if f'logits_{model_name}' in instance['sent_perturbs'][sent_id]:
                            pred = instance['sent_perturbs'][sent_id][f'logits_{model_name}']
                            dsid_sentid_newsent_predictions_sents[ds_id][sent_id] += [pred]

        for ds_id, sentences in dsid_sentid_newsent_predictions_sents.items():  # TODO sort negatives by disagr/conf
            for sent_id, predictions in sentences.items():
                instance = ds_id_instance[ds_id]
                preds = [np.argmax(predictions[i]) for i in range(len(predictions))]
                if instance['label'] != NEI_LABEL:
                    predicted_majority, count = Counter(preds).most_common(1)[0]
                    preds_avg = [np.mean([predictions[i][j] for i in range(len(predictions))]) for j in
                                 range(len(predictions[0]))]
                    if count >= args.min_nei_counts_sent and preds_avg[NEI_LABEL] >= args.min_nei_preds_sent and \
                            predicted_majority == NEI_LABEL and int(sent_id) < len(instance['text']):
                        if 'n_sent' in args.negatives_mode:
                            instance_new = copy.deepcopy(instance)
                            instance_new['text'] = instance_new['text'][:int(sent_id)] + instance_new['text'][int(sent_id)+1:]
                            instance_new['label'] = NEI_LABEL
                            instance_new['cl'] = 'neg'
                            instance_new['ds_id'] = ds_id
                            instance_new['pred_avg'] = preds_avg[NEI_LABEL]
                            new_sent_instances.append(instance_new)
                            additional_sent_negatives[ds_id].append(instance_new)

    added_pos = set()
    print(len(additional_sent_negatives),len(additional_negatives), len(additional_n_const_pos))
    instances_pos, instances_sent_neg = [], []
    additional_negatives_sampled = random.choices(additional_negatives,
                                                weights=[i['pred_avg'] for i in additional_negatives],
                                                k=min(len(additional_negatives), args.sample_instances))

    for instance in additional_negatives_sampled:
        ds_id = instance['ds_id']
        additional_instances += [instance]
        if ds_id not in added_pos:
            added_pos.add(ds_id)
            instances_pos += random.sample(additional_positives[str(ds_id)],min(args.max_positives, len(additional_positives[str(ds_id)])))

    # instances_sent_neg_sorted = sorted(new_sent_instances, key=lambda x: x['pred_avg'])[:args.sample_instances]
    if new_sent_instances:
        instances_sent_neg_samples = random.choices(new_sent_instances,
                                                    weights=[i['pred_avg'] for i in new_sent_instances],
                                                    k=min(len(additional_sent_negatives.keys()), args.sample_instances))
        for instance in instances_sent_neg_samples:
            instances_sent_neg.append(instance)
            if instance['ds_id'] not in added_pos:
                added_pos.add(instance['ds_id'] )
                instances_pos += random.sample(additional_positives[str(instance['ds_id'])], min(args.max_positives, len(additional_positives[str(instance['ds_id'])])))

    # for ds_id in random.sample(list(additional_sent_negatives.keys()), min(len(additional_sent_negatives.keys()), args.sample_instances)):
    #     instances_sent_neg += random.sample(additional_sent_negatives[str(ds_id)], min(args.max_negatives, len(additional_sent_negatives[str(ds_id)])))
    #     if ds_id not in added_pos:
    #         added_pos.add(ds_id)
    #         instances_pos += random.sample(additional_positives[str(ds_id)],
    #                                    min(args.max_positives, len(additional_positives[str(ds_id)])))
    insances_n_const_pos = []
    if 'n_sent_pos' in args.negatives_mode:

        additional_n_const_pos_sampled = random.choices(additional_n_const_pos,
                       weights=[i['overlap'] for i in additional_n_const_pos],
                       k=min(len(additional_n_const_pos), args.sample_instances))


        for instance in additional_n_const_pos_sampled:
            insances_n_const_pos += [instance]
            ds_id = instance['ds_id']
            if ds_id not in added_pos:
                added_pos.add(ds_id)
                instances_pos += random.sample(additional_positives[str(ds_id)],
                                           min(args.max_positives, len(additional_positives[str(ds_id)])))

    insances_nei = []
    if args.use_nei:
        for ds_id in random.sample(list(additional_nei.keys()), min(len(additional_nei), args.sample_instances)):
            insances_nei += random.sample(additional_nei[str(ds_id)],
                                                min(args.max_negatives, len(additional_n_const_pos[str(ds_id)])))

    print(len(additional_instances), len(instances_sent_neg), len(instances_pos), len(insances_n_const_pos), len(insances_nei))
    additional_instances += instances_sent_neg
    additional_instances += instances_pos
    additional_instances += insances_n_const_pos
    additional_instances += insances_nei

    print(f'Total added {len(additional_instances)}')
    return additional_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--run", help="Random seed", type=int, default=1)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=6)
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['liar_just', 'liar_ruling', 'fever', 'hover', 'fever_doc', 'fever_adv', 'vitaminc'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/', type=str)
    parser.add_argument("--test_dir", help="Path to the train datasets",
                        default='data/', type=str)

    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        type=str)
    parser.add_argument("--load_model_path",
                        help="Path where the model will be serialized",
                        type=str, nargs='+')
    parser.add_argument("--pretrained_path",
                        help="Path where the model will be serialized",
                        type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--accum_steps", help="Gradient accumulation steps", type=int, default=1)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-5)
    parser.add_argument("--max_len", help="Learning Rate", type=int,
                        default=512)
    parser.add_argument("--max_claim_len", help="Learning Rate", type=int,
                        default=512)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train', 'test', 'test_dev', 'data_map', 'test_from_file'])

    # Contrastive learning arguments
    parser.add_argument("--augment", help="Flag to include contrastive augmentation.",
                        action='store_true', default=False)

    parser.add_argument("--current_model", help="Temperature", type=str, default='bert')
    parser.add_argument("--other_models", help="Names of other models", type=str, default=['roberta', 'albert'],
                        nargs='+')
    parser.add_argument("--serialized_predictions_path", help="Path to the serialized predictions",
                        default=['data/fever_roberta_train_precomputed_predictions.json',
                                 'data/fever_albert_train_precomputed_predictions.json'], type=str, nargs='+')
    parser.add_argument("--serialized_sentence_predictions_path", help="Path to the serialized predictions",
                        default=['data/fever_roberta_train_precomputed_predictions.json',
                                 'data/fever_albert_train_precomputed_predictions.json'], type=str, nargs='+')
    parser.add_argument("--perturbations", help="Path to the precomputed perturbations of the positives",
                        default='data/pos_precomp_train_fever_word.json', type=str)

    parser.add_argument("--min_nei_preds", help="Minumum number of other models prediction NEI", type=float, default=2)
    parser.add_argument("--max_negatives", help="Maximim number of negativesI", type=int, default=2)
    parser.add_argument("--max_positives", help="Maximim number of positives", type=int, default=1)
    parser.add_argument("--use_nei", help="Flag for using NEI instances during training",
                        action='store_true', default=False)
    parser.add_argument("--positives_type", choices=['word', 'vector'], default='word')
    parser.add_argument("--positives_mode", choices=['p_sent', 'p_sent_const'], default=[], nargs='+')
    parser.add_argument("--negatives_mode", choices=['n_sent', 'n_const', 'n_sent_pos', 'n_const_pos'], default=[], nargs='+')
    parser.add_argument("--disagreements_mode", choices=['diff', 'same'], default=None, nargs='+')
    parser.add_argument("--sample_instances", help="Number of augmented instances to add to the training dataset", type=int, default=2000)
    parser.add_argument("--min_nei_counts", help="Minumum number of other models prediction NEI", type=int, default=3)
    parser.add_argument("--min_nei_counts_sent", help="Minumum number of other models prediction NEI", type=int, default=3)
    parser.add_argument("--min_nei_preds_sent", help="Minumum number of other models prediction NEI", type=float, default=2)

    args = parser.parse_args()
    seeds = [73, 13, 42]
    args.seed = seeds[args.run - 1]
    args.current_model = args.current_model + f'_{args.run}'
    args.model_path = f"models_test/{args.dataset}_{args.current_model}"
    if args.augment:
        args.model_path = args.model_path + '_cl_aug'
    args.model_path = args.model_path + f'_{args.run}'
    args.other_models = [f'roberta_{args.run}', f'bert_{args.run}', f'albert_{args.run}']
    args.serialized_predictions_path = [
        f"data/{args.dataset}_{args.other_models[i]}_train_precomputed_predictions.json"
        for i in range(len(args.other_models))
    ]
    print(args.serialized_predictions_path)
    args.perturbations = f"data/{args.dataset}_train_word_precomputed_perturbations.json"
    args.serialized_sentence_predictions_path  = [
        f"data/{args.dataset}_{args.other_models[i]}_train_precomputed_sentence_predictions.json"
        for i in range(len(args.other_models))
    ]

    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    NEI_LABEL = 1 if args.dataset == 'hover' else 2
    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    train, val, test = get_datasets(args.dataset_dir,
                                    args.dataset,
                                    args.labels)
    print(f'Train size {len(train)}', flush=True)
    print(f'Dev size {len(val)}', flush=True)

    print('Loaded data...', flush=True)
    if args.pretrained_path.startswith('bert'):
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path)
        config = BertConfig.from_pretrained(args.pretrained_path)
        config.num_labels = args.labels
        model = BertForSequenceClassification.from_pretrained(args.pretrained_path,
                                                              config=config).to(
            device)
    elif args.pretrained_path.startswith('roberta'):
        tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained_path)
        config = RobertaConfig.from_pretrained(args.pretrained_path)
        config.num_labels = args.labels
        model = RobertaForSequenceClassification.from_pretrained(
            args.pretrained_path,
            config=config).to(device)
    elif args.pretrained_path.startswith('albert'):
        tokenizer = AlbertTokenizerFast.from_pretrained(args.pretrained_path)
        config = AlbertConfig.from_pretrained(args.pretrained_path)
        config.num_labels = args.labels
        model = AlbertForSequenceClassification.from_pretrained(
            args.pretrained_path,
            config=config).to(device)

    collate_fn = partial(collate_explanations,
                         tokenizer=tokenizer,
                         device=device,
                         pad_to_max_length=True,
                         max_length=args.max_len,
                         max_claim_len=args.max_claim_len)
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      betas=(0.9, 0.98))

    if args.mode in ['test', 'test_dev']:
        test_dls = []
        if args.mode == 'test':
            ds_nei = NEIDataset(args.test_dir, nei_label_id=NEI_LABEL)
            test_dl_nei = DataLoader(ds_nei, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)
            test_dls.append(test_dl_nei)

            test_dl = DataLoader(test, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)
            test_dls.append(test_dl)
        else:
            test_dl = DataLoader(val, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)

        for test_dl in test_dls:
            results = []
            for model_path in args.load_model_path:
                checkpoint = torch.load(model_path)

                model.load_state_dict(checkpoint['model'])
                result = eval_model(args, model, test_dl)
                print(result, flush=True)
                results.append(result)

            for k in results[0].keys():
                mean = np.mean([results[i][k] for i in range(len(results))])
                std = np.std([results[i][k] for i in range(len(results))])
                print(k, f'{mean*100:.2f}', f'{std*100:.2f}')
    else:
        dev_dl = DataLoader(batch_size=args.batch_size,
                            dataset=val,
                            collate_fn=collate_fn,
                            shuffle=False)

        test_dl = DataLoader(test, batch_size=args.batch_size,
                             collate_fn=collate_fn, shuffle=False)

        ds_nei = NEIDataset(args.test_dir, nei_label_id=NEI_LABEL)
        test_dl_nei = DataLoader(ds_nei, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)

        if args.augment:
            additional_instances = add_precomputed(args, train, NEI_LABEL)
            train.dataset += additional_instances

        train_dl = DataLoader(batch_size=args.batch_size,
                              dataset=train, shuffle=True,
                              collate_fn=collate_fn)

        step_per_epoch = math.ceil(len(train) / (args.batch_size))
        num_steps = step_per_epoch * args.epochs

        best_model_w, best_perf = train_model(args, model,
                                              train_dl, dev_dl,
                                              optimizer, None)

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(best_perf)
        print(args)
        torch.save(checkpoint, args.model_path)

        model.load_state_dict(best_model_w)

        result = eval_model(args, model, test_dl)
        print('Test on test ds', result, flush=True)
        result = eval_model(args, model, test_dl_nei)
        print('Test on nei ds', result, flush=True)
