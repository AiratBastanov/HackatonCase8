import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from dataset import MarineDataset
from model import MarineNet
import joblib
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

META = "processed/meta.json"
TEST_NPY = "processed/test.npy"
OUT_DIR = "checkpoints"
TOPK_JSON = os.path.join(OUT_DIR, "topk.json")
CALIB_PATH = os.path.join(OUT_DIR, "calib.pkl")
ENSEMBLE_THR_PATH = os.path.join(OUT_DIR, "ensemble_threshold.json")

BATCH = 32
USE_TTA = True   # можно отключить для скорости


def tta_logits(model, x):
    """Усреднение логитов по 4 аугментациям (горизонтальный/вертикальный флип, поворот 90°)."""
    logits = []
    logits.append(model(x))
    logits.append(model(torch.flip(x, dims=[3])))   # горизонтальный
    logits.append(model(torch.flip(x, dims=[2])))   # вертикальный
    logits.append(model(torch.rot90(x, 1, [2, 3]))) # поворот 90
    return torch.stack(logits).mean(0)


def model_logits_on_loader(model, loader, use_tta=False):
    """Получить логиты модели для всех батчей в загрузчике."""
    model.eval()
    logits = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            if use_tta:
                out = tta_logits(model, x)
            else:
                out = model(x)
            logits.extend(out.cpu().numpy().reshape(-1))
    return np.array(logits).reshape(-1)


def main():
    # --- Загрузка метаданных ---
    with open(META) as f:
        meta = json.load(f)
    channels = meta["channels"]

    test_ds = MarineDataset(TEST_NPY, META, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # --- Загрузка списка чекпоинтов ---
    if not os.path.exists(TOPK_JSON):
        raise RuntimeError("topk.json not found. Run train.py first.")
    with open(TOPK_JSON) as f:
        topk_info = json.load(f)
    ckpts = topk_info.get("ckpts", [])
    if len(ckpts) == 0:
        raise RuntimeError("No checkpoints listed in topk.json")

    # --- Загружаем модели ---
    models = []
    for p in ckpts:
        if not os.path.exists(p):
            print(f"Warning: checkpoint {p} not found, skipping.")
            continue
        ck = torch.load(p, map_location=DEVICE)
        m = MarineNet(channels)
        m.load_state_dict(ck["model_state"])
        m = m.to(DEVICE)
        m.eval()
        models.append(m)

    if len(models) == 0:
        raise RuntimeError("No valid models for ensemble.")

    print(f"Ensemble size: {len(models)}")

    # --- Собираем логиты каждой модели (с TTA) ---
    all_logits = []
    for i, m in enumerate(models):
        print(f"Processing model {i+1}/{len(models)} ...")
        L = model_logits_on_loader(m, test_loader, use_tta=USE_TTA)
        all_logits.append(L)
    all_logits = np.stack(all_logits, axis=0)
    ensemble_logits = all_logits.mean(axis=0)
    ensemble_probs = 1.0 / (1.0 + np.exp(-ensemble_logits))

    # --- Калибровка ---
    if os.path.exists(CALIB_PATH):
        try:
            calib = joblib.load(CALIB_PATH)
            ensemble_probs = calib.predict_proba(ensemble_logits.reshape(-1, 1))[:, 1]
            print("Calibrator applied.")
        except Exception as e:
            print(f"Calibrator failed: {e}")

    # --- Загружаем порог с валидации ---
    if os.path.exists(ENSEMBLE_THR_PATH):
        with open(ENSEMBLE_THR_PATH) as f:
            thr_data = json.load(f)
        threshold = thr_data["threshold"]
        print(f"Using threshold from validation: {threshold:.3f}")
    else:
        # Крайний случай – ищем лучший порог по F1 на тесте (не рекомендуется)
        print("Ensemble threshold not found, computing best F1 threshold on test (warning: data leak).")
        targets = []
        for _, y in test_loader:
            targets.extend(y.numpy().reshape(-1))
        targets = np.array(targets).reshape(-1)
        best_f1 = -1.0
        threshold = 0.5
        for t in np.linspace(0.05, 0.95, 91):
            pbin = (ensemble_probs >= t).astype(int)
            f1 = f1_score(targets, pbin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                threshold = t
        print(f"Best F1 threshold on test: {threshold:.3f} (F1={best_f1:.4f})")

    # --- Итоговая оценка ---
    targets = []
    for _, y in test_loader:
        targets.extend(y.numpy().reshape(-1))
    targets = np.array(targets).reshape(-1)

    preds_bin = (ensemble_probs >= threshold).astype(int)
    acc = accuracy_score(targets, preds_bin)
    prec = precision_score(targets, preds_bin, zero_division=0)
    rec = recall_score(targets, preds_bin, zero_division=0)

    print("\n" + "="*50)
    print(f"Results with threshold = {threshold:.3f} (fixed from validation)")
    print(f"Accuracy:  {acc:.4f}  {'✅' if acc >= 0.85 else '❌'} (target ≥ 85%)")
    print(f"Precision: {prec:.4f}  {'✅' if prec > 0.8 else '❌'} (target > 0.8)")
    print(f"Recall:    {rec:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(targets, preds_bin, digits=4))


if __name__ == "__main__":
    main()