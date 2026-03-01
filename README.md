# Marine Debris Detector from Sentinel-2 Imagery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🏆 Хакатон "Детектор помех: отличи мусор от водорослей"

**Решение для кейса №8** – интеллектуальный классификатор, отличающий морской пластик от природных явлений (водоросли, пена, блики) по мультиспектральным снимкам Sentinel-2.

### 📊 Результаты

| Метрика | Значение | Требование |
|---------|----------|------------|
| Accuracy | **86.62%** | ≥85% ✅ |
| Precision (мусор) | **87.50%** | >0.8 ✅ |
| Recall (мусор) | 43.75% | – |

### 🎯 Особенности решения

- **Мультиспектральные данные**: работа с 12 каналами Sentinel-2 (9 исходных + 3 индекса: NDVI, NDWI, FDI)
- **Трансферное обучение**: адаптация EfficientNet-B0 (предобученной на ImageNet) под 12 каналов
- **Ансамбль**: усреднение трёх лучших чекпоинтов (по валидационному F1)
- **Калибровка**: Platt scaling для коррекции вероятностей
- **Умный выбор порога**: максимизация accuracy при условии precision ≥ 0.8
- **Test Time Augmentation**: 4 аугментации для повышения устойчивости
- **Лёгкая модель**: всего ~5 млн параметров, подходит для бортовых систем

### 📁 Структура проекта

marine-debris-detector/
marine-debris-detector/
├── preprocess.py # Загрузка и предобработка датасетов
├── dataset.py # PyTorch Dataset с аугментациями
├── model.py # Архитектура модели (EfficientNet-B0 + SE + SPP)
├── train.py # Скрипт обучения с ансамблем и калибровкой
├── eval.py # Оценка модели на тесте
├── requirements.txt # Зависимости
├── README.md # Документация
└── data/ # Папка с датасетами (создаётся пользователем)
├── MARIDA/ # MARIDA dataset
│ └── patches/ # Патчи MARIDA
├── MADOS/ # MADOS dataset
└── PLP2019_dataset/ # PLP2019 dataset
### 📥 Используемые датасеты

1. **[MARIDA](https://zenodo.org/records/5151941)** – патчи с масками мусора
2. **[MADOS](https://zenodo.org/records/10664073)** – сцены с масками мусора
3. **[PLP2019](https://zenodo.org/records/3752719)** – искусственные мишени с координатами

### 🚀 Быстрый старт

#### Установка зависимостей

bash
pip install -r requirements.txt

#### Подготовка данных
Скачайте датасеты и разместите их в папке data/ согласно структуре выше

Запустите предобработку:
python preprocess.py
#### Обучение
Модель будет сохранять:

Топ-3 чекпоинта по валидационному F1 в папку checkpoints/

best_f1.pt – модель с лучшим F1

best_prec.pt – модель, достигающая precision ≥ 0.8

calib.pkl – калибратор для ансамбля

ensemble_threshold.json – порог для ансамбля (precision ≥ 0.8 с макс. accuracy)

#### Оценка
python eval.py

#### Визуализация результатов
Для построения графиков запустите:
python visualize.py

Будут созданы:

figures/confusion_matrix.png

figures/roc_curve.png

figures/pr_curve.png

figures/prob_hist.png

#### Технические детали
Архитектура модели
Input (12×256×256)
    ↓
    Adapted ConvStem (из EfficientNet-B0, 3 → 12 каналов)
    ↓
    EfficientNet-B0 backbone (features_only=True)
    ↓
    SEBlock (внимание по каналам)
    ↓
    SpatialPyramidPooling (1×1, 2×2, 4×4)
    ↓
    AdaptiveAvgPool2d(1)
    ↓
    FC (feature_channels → 256) → BN → ReLU → Dropout → FC (256 → 1)
    ↓
    Output (логит)

#### Обучение
Loss: Focal Loss (γ=2.0) для борьбы с дисбалансом классов

Сэмплинг: WeightedRandomSampler для балансировки батчей

Оптимизатор: AdamW (lr=1e-4, weight_decay=1e-5)

Scheduler: CosineAnnealingLR (T_max=EPOCHS)

Эпох: 50

Batch size: 32
