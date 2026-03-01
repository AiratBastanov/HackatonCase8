"""
Скрипт обучения.
Сохраняет top-K чекпоинтов (по val F1), собирает валидационные логиты ансамбля,
обучает Platt-калибровку (логистическая регрессия), сохраняет best_f1.pt и best_prec.pt,
а также калибратор и порог для ансамбля, обеспечивающий precision ≥ 0.8 с максимальной accuracy.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from dataset import MarineDataset
from model import MarineNet
import random
import joblib
from sklearn.linear_model import LogisticRegression

SEED = 42
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
META_PATH = "processed/meta.json"
TRAIN_NPY = "processed/train.npy"
VAL_NPY = "processed/val.npy"
TEST_NPY = "processed/test.npy"
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

BEST_F1_PATH = os.path.join(OUT_DIR, "best_f1.pt")
BEST_PREC_PATH = os.path.join(OUT_DIR, "best_prec.pt")
CALIB_PATH = os.path.join(OUT_DIR, "calib.pkl")
ENSEMBLE_THR_PATH = os.path.join(OUT_DIR, "ensemble_threshold.json")

TOP_K = 3
PRECISION_TARGET = 0.80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом классов."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.view(-1)
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def find_thresholds(preds, targets):
    """Поиск оптимальных порогов по F1 и по precision."""
    best_thr_f1 = 0.5
    best_f1 = -1.0
    best_thr_prec = None
    best_prec = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pbin = (preds >= t).astype(int)
        f = f1_score(targets, pbin)
        prec = precision_score(targets, pbin, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr_f1 = t
        if prec > best_prec:
            best_prec = prec
            best_thr_prec = t

    thr_prec_target = None
    prec_target_recall = -1.0
    if best_thr_prec is not None:
        for t in np.linspace(0.05, 0.95, 91):
            pbin = (preds >= t).astype(int)
            prec = precision_score(targets, pbin, zero_division=0)
            rec = recall_score(targets, pbin, zero_division=0)
            if prec >= PRECISION_TARGET and rec > prec_target_recall:
                thr_prec_target = t
                prec_target_recall = rec
    return best_thr_f1, best_f1, thr_prec_target, best_prec, best_thr_prec


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    with open(META_PATH) as f:
        meta = json.load(f)
    channels = meta["channels"]

    train_ds = MarineDataset(TRAIN_NPY, META_PATH, augment=True)
    val_ds = MarineDataset(VAL_NPY, META_PATH, augment=False)

    train_labels = np.array([int(x[1]) for x in train_ds.data])
    class_sample_count = np.array([(train_labels == t).sum() for t in [0, 1]])
    weights = 1.0 / (class_sample_count + 1e-12)
    samples_weight = np.array([weights[int(l)] for l in train_labels])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = MarineNet(channels).to(DEVICE)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    top_k = []
    best_f1 = 0.0
    best_prec_overall = 0.0
    scaler = torch.amp.GradScaler('cuda') if DEVICE.startswith("cuda") else None
    saved_models_info = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch}")
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            running_loss += float(loss.item())
        scheduler.step()

        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x)
                val_logits.extend(out.cpu().numpy().reshape(-1))
                val_targets.extend(y.numpy().reshape(-1))
        val_logits = np.array(val_logits).reshape(-1)
        val_targets = np.array(val_targets).reshape(-1)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        best_thr_f1, best_f1_local, thr_prec_target, best_prec_local, _ = find_thresholds(val_probs, val_targets)

        pbin = (val_probs >= best_thr_f1).astype(int)
        acc = accuracy_score(val_targets, pbin)
        prec = precision_score(val_targets, pbin, zero_division=0)
        rec = recall_score(val_targets, pbin, zero_division=0)
        print(f"best_thr_f1: {best_thr_f1:.3f}  val F1: {best_f1_local:.4f}  acc: {acc:.4f}  prec: {prec:.4f}  rec: {rec:.4f}")

        ckpt_path = os.path.join(OUT_DIR, f"epoch_{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)
        saved_models_info.append({"path": ckpt_path, "val_logits": val_logits.copy(), "val_targets": val_targets.copy(), "f1": float(best_f1_local)})

        top_k.append((float(best_f1_local), ckpt_path))
        top_k = sorted(top_k, key=lambda x: x[0], reverse=True)[:TOP_K]

        if best_f1_local > best_f1:
            best_f1 = best_f1_local
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "threshold": float(best_thr_f1)}, BEST_F1_PATH)
            print("Saved model (best F1)")

        if thr_prec_target is not None:
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "threshold": float(thr_prec_target)}, BEST_PREC_PATH)
            print(f"Saved model (meets precision target {PRECISION_TARGET}) at thr {thr_prec_target:.3f}")

        if best_prec_local > best_prec_overall:
            best_prec_overall = best_prec_local

        print(f"epoch loss: {running_loss/len(train_loader):.4f} | best_f1 so far: {best_f1:.4f} | best_prec seen: {best_prec_overall:.4f}")

    # --- после обучения: ансамбль и калибратор ---
    print("\nTraining finished. Preparing ensemble and calibrator...")
    top_ckpts = [p for _, p in top_k]
    print("Top-K checkpoints:", top_ckpts)

    path2logits = {info["path"]: info["val_logits"] for info in saved_models_info}
    available = [p for p in top_ckpts if p in path2logits]

    if len(available) == 0:
        print("No val logits available for top checkpoints — skipping calibrator.")
    else:
        stacked = np.stack([path2logits[p] for p in available], axis=0)
        ensemble_val_logits = stacked.mean(axis=0)
        ensemble_val_probs = 1.0 / (1.0 + np.exp(-ensemble_val_logits))
        val_targets = saved_models_info[0]["val_targets"]

        # Platt scaling
        clf = LogisticRegression(solver='lbfgs')
        try:
            clf.fit(ensemble_val_logits.reshape(-1, 1), val_targets)
            joblib.dump(clf, CALIB_PATH)
            print("Saved calibrator to", CALIB_PATH)
        except Exception as e:
            print("Calibrator training failed:", e)

        # Поиск порога для ансамбля, дающего precision >= target с максимальной accuracy
        best_thr_ensemble = None
        best_acc_ensemble = -1.0
        for t in np.linspace(0.05, 0.95, 91):
            pbin = (ensemble_val_probs >= t).astype(int)
            prec = precision_score(val_targets, pbin, zero_division=0)
            acc = accuracy_score(val_targets, pbin)
            if prec >= PRECISION_TARGET and acc > best_acc_ensemble:
                best_acc_ensemble = acc
                best_thr_ensemble = t
        if best_thr_ensemble is not None:
            with open(ENSEMBLE_THR_PATH, "w") as f:
                json.dump({"threshold": best_thr_ensemble}, f)
            print(f"Ensemble threshold for precision>={PRECISION_TARGET} (max acc) saved: {best_thr_ensemble:.3f}")
        else:
            print("No threshold on validation meets precision target for ensemble.")

    # Сохраняем список топовых чекпоинтов
    top_info = {"ckpts": top_ckpts, "channels": channels}
    with open(os.path.join(OUT_DIR, "topk.json"), "w") as f:
        json.dump(top_info, f, indent=2)

    print("Top-K saved and topk.json written.")
    print("Best F1:", best_f1)
    print("Best precision seen (any threshold):", best_prec_overall)


if __name__ == "__main__":
    main()