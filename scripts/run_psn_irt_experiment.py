#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate PSN-IRT on explicit train/val/test interaction splits."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PSNIRT(nn.Module):
    def __init__(self, num_students, num_items, hidden_dim=64, model_type="4PL"):
        super().__init__()
        self.model_type = model_type.upper()
        output_dims = {"1PL": 1, "2PL": 2, "3PL": 3, "4PL": 4}
        if self.model_type not in output_dims:
            raise ValueError("model_type must be one of 1PL, 2PL, 3PL, 4PL")

        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dims[self.model_type]),
        )

    def forward(self, student_onehot, item_onehot):
        theta = self.student_net(student_onehot)
        params = self.item_net(item_onehot)

        b = params[:, 0].unsqueeze(1)
        a = torch.ones_like(theta)
        c = torch.zeros_like(theta)
        d = torch.ones_like(theta)

        if self.model_type in {"2PL", "3PL", "4PL"}:
            a = params[:, 1].unsqueeze(1)
        if self.model_type in {"3PL", "4PL"}:
            c = torch.sigmoid(params[:, 2].unsqueeze(1))
        if self.model_type == "4PL":
            d = torch.sigmoid(params[:, 3].unsqueeze(1))

        prob = c + (d - c) * torch.sigmoid(a * (theta - b))
        return prob.clamp(1e-6, 1.0 - 1e-6)


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.num_students = df.shape[0]
        self.num_items = df.shape[1]
        values = df.to_numpy()
        rows, cols = np.where(~pd.isna(values))
        labels = values[rows, cols].astype(np.float32)
        self.student_ids = torch.tensor(rows, dtype=torch.long)
        self.item_ids = torch.tensor(cols, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.student_ids[idx], self.item_ids[idx], self.labels[idx]


def collate_interactions(batch, num_students, num_items):
    student_ids, item_ids, labels = zip(*batch)
    student_ids = torch.stack(student_ids)
    item_ids = torch.stack(item_ids)
    labels = torch.stack(labels)

    student_onehot = torch.zeros(len(batch), num_students, dtype=torch.float32)
    item_onehot = torch.zeros(len(batch), num_items, dtype=torch.float32)
    student_onehot[torch.arange(len(batch)), student_ids] = 1.0
    item_onehot[torch.arange(len(batch)), item_ids] = 1.0
    return student_onehot, item_onehot, labels, student_ids, item_ids


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_interactions(batch, dataset.num_students, dataset.num_items),
    )


def evaluate(model, loader, device):
    model.eval()
    labels, probs, student_ids, item_ids = [], [], [], []
    with torch.no_grad():
        for student, item, y, sid, qid in loader:
            pred = model(student.to(device), item.to(device)).squeeze(1).cpu()
            labels.extend(y.numpy())
            probs.extend(pred.numpy())
            student_ids.extend(sid.numpy())
            item_ids.extend(qid.numpy())

    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs > 0.5).astype(int)
    bce = -np.mean(labels * np.log(np.clip(probs, 1e-6, 1.0 - 1e-6)) +
                   (1.0 - labels) * np.log(np.clip(1.0 - probs, 1e-6, 1.0 - 1e-6)))
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5
    return {
        "loss": float(bce),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(auc),
    }, pd.DataFrame({
        "student_id": student_ids,
        "item_id": item_ids,
        "ground_truth": labels.astype(int),
        "prediction": preds,
        "probability": probs,
    })


def save_parameters(model, num_students, num_items, output_dir, device):
    model.eval()
    abilities = []
    with torch.no_grad():
        for sid in range(num_students):
            student = torch.zeros(1, num_students, dtype=torch.float32, device=device)
            student[0, sid] = 1.0
            abilities.append(model.student_net(student).cpu().item())

    pd.DataFrame({"student_id": range(num_students), "ability": abilities}).to_csv(
        output_dir / "student_abilities.csv", index=False
    )

    rows = []
    with torch.no_grad():
        for qid in range(num_items):
            item = torch.zeros(1, num_items, dtype=torch.float32, device=device)
            item[0, qid] = 1.0
            raw = model.item_net(item).cpu()[0]
            row = {"item_id": qid, "difficulty": raw[0].item()}
            if model.model_type in {"2PL", "3PL", "4PL"}:
                row["discriminability"] = raw[1].item()
            if model.model_type in {"3PL", "4PL"}:
                row["guessing"] = torch.sigmoid(raw[2]).item()
            if model.model_type == "4PL":
                row["feasibility"] = torch.sigmoid(raw[3]).item()
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "item_parameters.csv", index=False)


def save_training_curves(history, output_dir):
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped training_curves.png")
        return

    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train_loss")
    if "loss" in history_df.columns:
        axes[0].plot(history_df["epoch"], history_df["loss"], marker="o", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCELoss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for metric in ["acc", "f1", "auc"]:
        axes[1].plot(history_df["epoch"], history_df[metric], marker="o", label=f"val_{metric}")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def init_wandb(args):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Run `uv pip install wandb` or install requirements again."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/splits/paper_seed3407/train.csv")
    parser.add_argument("--val_csv", default="data/splits/paper_seed3407/val.csv")
    parser.add_argument("--test_csv", default="data/splits/paper_seed3407/test.csv")
    parser.add_argument("--output_dir", default="results/paper_reproduction/psn_irt_4pl_seed3407")
    parser.add_argument("--model_type", default="4PL", choices=["1PL", "2PL", "3PL", "4PL"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wandb_project", default=None, help="Enable W&B logging with this project name.")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    wandb_run = init_wandb(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv, header=None)
    val_df = pd.read_csv(args.val_csv, header=None)
    test_df = pd.read_csv(args.test_csv, header=None)

    train_data = InteractionDataset(train_df)
    val_data = InteractionDataset(val_df)
    test_data = InteractionDataset(test_df)
    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    val_loader = make_loader(val_data, args.batch_size, shuffle=False)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSNIRT(
        num_students=train_data.num_students,
        num_items=train_data.num_items,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    best_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for student, item, y, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}"):
            optimizer.zero_grad()
            pred = model(student.to(device), item.to(device)).squeeze(1)
            loss = criterion(pred, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_loader), **val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": row["train_loss"],
                "val/loss": row["loss"],
                "val/acc": row["acc"],
                "val/f1": row["f1"],
                "val/auc": row["auc"],
            }, step=epoch)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    save_training_curves(history, output_dir)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics, predictions = evaluate(model, test_loader, device)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    save_parameters(model, train_data.num_students, train_data.num_items, output_dir, device)
    if wandb_run is not None:
        import wandb

        wandb_run.log({
            "test/loss": test_metrics["loss"],
            "test/acc": test_metrics["acc"],
            "test/f1": test_metrics["f1"],
            "test/auc": test_metrics["auc"],
        })
        curves_path = output_dir / "training_curves.png"
        if curves_path.exists():
            wandb_run.log({"training_curves": wandb.Image(str(curves_path))})
        wandb_run.finish()
    print(json.dumps({"test": test_metrics, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
