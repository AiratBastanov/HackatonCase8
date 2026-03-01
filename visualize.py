import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from dataset import MarineDataset
from model import MarineNet
from torch.utils.data import DataLoader
import joblib
import os

# --- Конфигурация ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META = "processed/meta.json"
TEST_NPY = "processed/test.npy"
OUT_DIR = "checkpoints"
TOPK_JSON = os.path.join(OUT_DIR, "topk.json")
CALIB_PATH = os.path.join(OUT_DIR, "calib.pkl")
ENSEMBLE_THR_PATH = os.path.join(OUT_DIR, "ensemble_threshold.json")
BATCH = 32
USE_TTA = True
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --- Загрузка метаданных ---
with open(META) as f:
    meta = json.load(f)
channels = meta["channels"]

# --- Загрузка тестового датасета ---
test_ds = MarineDataset(TEST_NPY, META, augment=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# --- Загрузка топ-чекпоинтов ---
with open(TOPK_JSON) as f:
    topk_info = json.load(f)
ckpts = topk_info["ckpts"]

def tta_logits(model, x):
    logits = [model(x)]
    logits.append(model(torch.flip(x, dims=[3])))   # горизонтальный
    logits.append(model(torch.flip(x, dims=[2])))   # вертикальный
    logits.append(model(torch.rot90(x, 1, [2, 3]))) # поворот 90
    return torch.stack(logits).mean(0)

def model_logits_on_loader(model, loader, use_tta=False):
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

# Загружаем модели
models = []
for p in ckpts:
    if not os.path.exists(p):
        continue
    ck = torch.load(p, map_location=DEVICE)
    m = MarineNet(channels)
    m.load_state_dict(ck["model_state"])
    m = m.to(DEVICE)
    m.eval()
    models.append(m)

if len(models) == 0:
    raise RuntimeError("No models loaded.")

# Получаем логиты каждой модели (с TTA)
all_logits = []
for m in models:
    L = model_logits_on_loader(m, test_loader, use_tta=USE_TTA)
    all_logits.append(L)
all_logits = np.stack(all_logits)  # [K, N]
ensemble_logits = all_logits.mean(axis=0)
ensemble_probs = 1.0 / (1.0 + np.exp(-ensemble_logits))

# Применяем калибратор
if os.path.exists(CALIB_PATH):
    calib = joblib.load(CALIB_PATH)
    ensemble_probs = calib.predict_proba(ensemble_logits.reshape(-1, 1))[:, 1]

# Истинные метки
targets = []
for _, y in test_loader:
    targets.extend(y.numpy().reshape(-1))
targets = np.array(targets).reshape(-1)

# --- Порог из валидации ---
if os.path.exists(ENSEMBLE_THR_PATH):
    with open(ENSEMBLE_THR_PATH) as f:
        thr_data = json.load(f)
    threshold = thr_data["threshold"]
else:
    threshold = 0.5  # fallback

preds_bin = (ensemble_probs >= threshold).astype(int)

# --- 1. Матрица ошибок ---
cm = confusion_matrix(targets, preds_bin)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Вода', 'Мусор'], yticklabels=['Вода', 'Мусор'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title(f'Матрица ошибок (порог={threshold:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# --- 2. ROC-кривая ---
fpr, tpr, _ = roc_curve(targets, ensemble_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (ансамбль)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curve.png'), dpi=150)
plt.close()

# --- 3. Precision-Recall кривая ---
precision, recall, thresholds_pr = precision_recall_curve(targets, ensemble_probs)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label='PR curve', linewidth=2)
plt.scatter(recall[np.argmin(np.abs(thresholds_pr - threshold))],
            precision[np.argmin(np.abs(thresholds_pr - threshold))],
            color='red', s=80, zorder=5, label=f'порог={threshold:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'pr_curve.png'), dpi=150)
plt.close()

# --- 4. Гистограмма распределения вероятностей ---
plt.figure(figsize=(8,5))
plt.hist(ensemble_probs[targets==0], bins=50, alpha=0.7, label='Вода', color='blue')
plt.hist(ensemble_probs[targets==1], bins=50, alpha=0.7, label='Мусор', color='red')
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'порог {threshold:.3f}')
plt.xlabel('Предсказанная вероятность класса "мусор"')
plt.ylabel('Количество образцов')
plt.title('Распределение предсказанных вероятностей')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'prob_hist.png'), dpi=150)
plt.close()

# --- 5. График обучения (загрузить из лога, если есть) ---
# Можно сохранить историю обучения в train.py, но для презентации можно вставить скриншот из логов.
# Вместо этого можно сгенерировать пример (синтетический) или просто указать, что данные доступны.

print(f"Графики сохранены в папку {FIG_DIR}")