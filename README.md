# Dataset
The **SufficientFacts** diagnostic test dataset can be found in:

* [data/sufficient_facts](data/sufficient_facts) directory 
* through the [HuggingFace datasets hub](https://huggingface.co/datasets/pepa/sufficient_facts)

Please, consult the README in the corresponding locations for more information on the dataset. 

# Contrastive Training Experiments

Example script for supervised training:
```bash
python3.8 modeling/train.py --gpu --dataset vitaminc --dataset_dir data/vitaminc --model_path vitaminc_roberta_1e5_1 --lr 1e-5 --pretrained_path 'roberta-base' --labels 3 --epochs 3 --max_len 256 --batch_size 16 --test_dir data/vitaminc_const.jsonl
```

Example script for CAD training:
```bash
python3.8 modeling/train.py --gpu --dataset vitaminc --dataset_dir data/vitaminc --lr 1e-5 --pretrained_path 'albert-base-v2' --labels 3 --current_model albert --negatives_mode n_const n_sent_pos --positives_mode p_sent p_sent_const --max_negatives 2 --max_positives 1 --test_dir data/vitaminc_const_rem.jsonl --min_nei_counts 2 --min_nei_preds 0.5 --min_nei_counts_sent 2 --min_nei_preds_sent 0.5 --max_len 256 --batch_size 16 --epochs 3 --sample_instances 20000
```

Example script for CL loss training:
```bash
python3.8 modeling/train_contrastive_loss.py --gpu --dataset vitaminc --dataset_dir data/vitaminc --lr 1e-5 --pretrained_path 'albert-base-v2' --labels 3 --current_model albert --negatives_mode n_const n_sent_pos --positives_mode p_sent p_sent_const --max_negatives 2 --max_positives 1 --test_dir data/vitaminc_const_rem.jsonl --min_nei_counts 2 --min_nei_preds 0.5 --min_nei_counts_sent 2 --min_nei_preds_sent 0.5 --max_len 256 --batch_size 16 --epochs 3 --sample_instances 20000 --temp 1.5
```