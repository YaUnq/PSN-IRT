# PSN-IRT Reproduction

This is an **unofficial reproduction repository** for the paper
*Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item
Response Theory*.

The goal of this repository is to reproduce the main PSN-IRT experimental
pipeline according to the paper setting, using the released binary response
matrix and a paper-style model-item interaction split. It is not the official
repository and is not maintained by the paper authors.

Original paper:
[arXiv:2505.15055](https://arxiv.org/abs/2505.15055)

Official repository:
[Joe-Hall-Lee/PSN-IRT](https://github.com/Joe-Hall-Lee/PSN-IRT)

## What This Repo Reproduces

The paper studies LLM benchmark quality with Item Response Theory. The core
pipeline is:

```text
12 LLMs x 41,871 benchmark items
        -> binary response matrix
        -> 60/20/20 model-item interaction split
        -> train PSN-IRT with a 4PL IRT objective
        -> evaluate held-out response prediction
        -> estimate model abilities and item parameters
        -> compute item-quality diagnostics with Fisher information
```

This repository focuses on the reproducible PSN-IRT part:

- paper-style train/validation/test split over model-item interactions
- PSN-IRT 4PL training and held-out prediction evaluation
- ACC, F1, AUC, and BCE loss logging
- model ability and item parameter export
- item-quality ranking by Fisher information
- optional Weights & Biases monitoring

## Repository Layout

```text
data/
  combine.csv                         # 12 x 41871 binary response matrix
  splits/paper_seed3407/              # generated 60/20/20 interaction split

scripts/
  split_interactions.py               # build paper-style interaction splits
  run_psn_irt_experiment.py           # train/evaluate PSN-IRT
  analyze_item_quality.py             # compute Fisher-information rankings

results/
  paper_reproduction/                 # generated experiment outputs

requirements.txt
README.md
```

## Environment

This project is intended to be run with `uv`.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you want to use Weights & Biases:

```bash
wandb login
```

## Step 1: Generate Interaction Splits

The paper describes splitting over model-item interactions rather than splitting
by benchmark or by item. Each observed cell in the response matrix is treated as:

```text
model_id, item_id, response
```

where `response` is `1` for correct and `0` for incorrect.

Generate the 60/20/20 split:

```bash
python scripts/split_interactions.py \
  --input_csv data/combine.csv \
  --output_dir data/splits/paper_seed3407 \
  --seed 3407
```

Expected split sizes:

```text
train: 301471 interactions
val:   100490 interactions
test:  100491 interactions
```

The generated files keep the original matrix shape and mask held-out
interactions with `NaN`.

## Step 2: Train PSN-IRT

Train the 4PL PSN-IRT model:

```bash
python scripts/run_psn_irt_experiment.py \
  --train_csv data/splits/paper_seed3407/train.csv \
  --val_csv data/splits/paper_seed3407/val.csv \
  --test_csv data/splits/paper_seed3407/test.csv \
  --output_dir results/paper_reproduction/psn_irt_4pl_seed3407 \
  --model_type 4PL \
  --max_epochs 50 \
  --batch_size 512 \
  --learning_rate 0.003 \
  --weight_decay 0.0001
```

With Weights & Biases:

```bash
python scripts/run_psn_irt_experiment.py \
  --train_csv data/splits/paper_seed3407/train.csv \
  --val_csv data/splits/paper_seed3407/val.csv \
  --test_csv data/splits/paper_seed3407/test.csv \
  --output_dir results/paper_reproduction/psn_irt_4pl_seed3407 \
  --model_type 4PL \
  --max_epochs 50 \
  --batch_size 512 \
  --learning_rate 0.003 \
  --weight_decay 0.0001 \
  --wandb_project psn-irt-reproduction \
  --wandb_run_name psn-irt-4pl-seed3407
```

For offline W&B logging:

```bash
--wandb_project psn-irt-reproduction --wandb_mode offline
```

Training outputs:

```text
results/paper_reproduction/psn_irt_4pl_seed3407/
  best_model.pt
  metrics.json
  test_predictions.csv
  training_history.csv
  training_curves.png
  student_abilities.csv
  item_parameters.csv
```

Logged metrics include:

```text
train/loss
val/loss
val/acc
val/f1
val/auc
test/loss
test/acc
test/f1
test/auc
```

## Step 3: Analyze Item Quality

After training, compute item-level quality diagnostics:

```bash
python scripts/analyze_item_quality.py \
  --item_parameters results/paper_reproduction/psn_irt_4pl_seed3407/item_parameters.csv \
  --student_abilities results/paper_reproduction/psn_irt_4pl_seed3407/student_abilities.csv \
  --output_csv results/paper_reproduction/psn_irt_4pl_seed3407/item_quality.csv \
  --selection_ratio 0.7
```

This produces:

```text
item_quality.csv
selected_item_indices_0.7.csv
```

`item_quality.csv` ranks all items by average Fisher information. The selected
indices file contains the top items under the requested selection ratio.

## Notes on Reproduction Scope

This repository is designed to reproduce the core released-data PSN-IRT
experiment, but it is not a full one-to-one reproduction of every table and
analysis in the paper.

Known limitations:

- item-to-benchmark metadata is not included in the released response matrix
- traditional IRT baselines such as MLE, MCMC, VI, and VIBO are not reproduced
  here
- benchmark-level aggregation requires additional metadata beyond
  `data/combine.csv`
- exact numbers may differ due to hardware, random seed behavior, and training
  implementation details

## Citation

If you use this reproduction, please cite the original paper:

```bibtex
@inproceedings{zhou2025lost,
      title={Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory},
      author={Zhou, Hongli and Huang, Hui and Zhao, Ziqing and Han, Lvyuan and Wang, Huicheng and Chen, Kehai and Yang, Muyun and Bao, Wei and Dong, Jian and Xu, Bing and Zhu, Conghui and Cao, Hailong and Zhao, Tiejun},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2026}
}
```

## Disclaimer

This project is an independent, unofficial reproduction effort. Any mistakes in
the code, configuration, or interpretation are the responsibility of this
repository, not the original paper authors.
