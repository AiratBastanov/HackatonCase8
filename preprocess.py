"""
Создает processed/train.npy, processed/val.npy, processed/test.npy и processed/meta.json.
Работает с MARIDA (patches), MADOS (Scene_*/10, Scene_*/20) и PLP2019.
Binary label: (mask == 1) -> presence of marine debris in patch.
"""
import os
import json
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import netCDF4 as nc
import fiona
from rasterio.features import rasterize
from shapely.geometry import shape
import re

BASE = "data"
MARIDA_PATCHES = os.path.join(BASE, "patches")
MADOS_ROOT = os.path.join(BASE, "MADOS")
PLP2019_ROOT = os.path.join(BASE, "PLP2019_dataset")

OUT = "processed"
os.makedirs(OUT, exist_ok=True)

TARGET_SIZE = (256, 256)
IMG_CLIP = (-1, 1)

BANDS_10 = ["492", "560", "665", "833"]
BANDS_20 = ["704", "783", "865", "1614", "2202"]

EXPECTED_CHANNELS = 12
PLASTIC_THRESHOLD = 0.01

RSEED = 42
random.seed(RSEED)
np.random.seed(RSEED)


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def read_nc(nc_path, target_wavelengths):
    """
    Чтение .nc файла ACOLITE.
    target_wavelengths: список целевых длин волн (например, [492, 560, 665, 704, 740, 783, 833, 1614, 2202]).
    Для каждой целевой длины волны ищется переменная rhos_* с ближайшей доступной длиной волны.
    Возвращает img (C, H, W) в порядке target_wavelengths, а также lat, lon.
    """
    ds = nc.Dataset(nc_path, 'r')
    lat = None
    lon = None

    for lat_name in ['lat', 'latitude', 'Lat', 'Latitude']:
        if lat_name in ds.variables:
            lat = ds.variables[lat_name][:]
            break
    for lon_name in ['lon', 'longitude', 'Lon', 'Longitude']:
        if lon_name in ds.variables:
            lon = ds.variables[lon_name][:]
            break

    # Собираем все переменные rhos_* с их длинами волн
    available = {}
    for vname in ds.variables:
        if vname.startswith('rhos_'):
            try:
                wl = float(vname.split('_')[1])
                available[wl] = vname
            except:
                continue

    if not available:
        raise KeyError(f"No rhos_* variables found in {nc_path}")

    img_channels = []
    for target in target_wavelengths:
        # Находим ближайшую доступную длину волны
        closest_wl = min(available.keys(), key=lambda x: abs(x - target))
        var_name = available[closest_wl]
        var = ds.variables[var_name]
        data = var[:].squeeze()
        img_channels.append(data.astype(np.float32))

    ds.close()
    img = np.stack(img_channels, axis=0)
    return img, lat, lon


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """Ресайз изображения (C, H, W) до size (H', W')."""
    c, h, w = img.shape
    if (h, w) == size:
        return img
    out = []
    for i in range(c):
        out.append(cv2.resize(img[i], size[::-1], interpolation=interpolation))
    return np.stack(out)


def indices(b):
    # b: (9, H, W) в порядке: B2, B3, B4, B5, B6, B7, B8, B11, B12
    B2, B3, B4 = b[0], b[1], b[2]
    B8 = b[5]
    B11 = b[7]
    eps = 1e-6
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    fdi = B8 - (B11 + B4)
    return np.stack([ndvi, ndwi, fdi]).astype(np.float32)


def load_marida():
    data = []
    files = []
    for root, _, f in os.walk(MARIDA_PATCHES):
        for x in f:
            if x.endswith(".tif") and "_cl" not in x and "_conf" not in x:
                files.append(os.path.join(root, x))
    print("MARIDA patches:", len(files))
    for p in tqdm(files, desc="MARIDA"):
        mask_path = p.replace(".tif", "_cl.tif")
        if not os.path.exists(mask_path):
            continue
        img = read_tif(p)
        mask = read_tif(mask_path)[0]
        try:
            img = img[[1, 2, 3, 4, 5, 7, 8, 9, 10]]
        except Exception:
            continue
        img = img / 10000.0
        img = np.nan_to_num(img)
        ind = indices(img)
        img = np.concatenate([img, ind], axis=0)
        if img.shape[0] != EXPECTED_CHANNELS:
            continue
        img = np.clip(img, *IMG_CLIP)
        img = resize(img, TARGET_SIZE)
        label = int((mask == 1).any())
        data.append((img.astype(np.float32), label))
    print("MARIDA samples:", len(data))
    return data


def load_mados():
    data = []
    if not os.path.exists(MADOS_ROOT):
        return data
    scenes = [x for x in os.listdir(MADOS_ROOT) if "Scene" in x]
    for scene in tqdm(scenes, desc="MADOS"):
        d10 = os.path.join(MADOS_ROOT, scene, "10")
        d20 = os.path.join(MADOS_ROOT, scene, "20")
        if not os.path.exists(d10):
            continue
        crops = []
        for f in os.listdir(d10):
            if "_rhorc_492_" in f:
                crops.append(f.split("_")[-1].replace(".tif", ""))
        for c in crops:
            bands = []
            ok = True
            for b in BANDS_10:
                p = os.path.join(d10, f"{scene}_L2R_rhorc_{b}_{c}.tif")
                if not os.path.exists(p):
                    ok = False
                    break
                bands.append(read_tif(p)[0])
            if not ok:
                continue
            shape = bands[0].shape
            for b in BANDS_20:
                p = os.path.join(d20, f"{scene}_L2R_rhorc_{b}_{c}.tif")
                if not os.path.exists(p):
                    ok = False
                    break
                img20 = read_tif(p)[0]
                img20 = cv2.resize(img20, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                bands.append(img20)
            if not ok:
                continue
            img = np.stack(bands).astype(np.float32)
            img = np.nan_to_num(img)
            ind = indices(img)
            img = np.concatenate([img, ind], axis=0)
            if img.shape[0] != EXPECTED_CHANNELS:
                continue
            img = np.clip(img, *IMG_CLIP)
            img = resize(img, TARGET_SIZE)
            mask_path = os.path.join(d10, f"{scene}_L2R_cl_{c}.tif")
            if not os.path.exists(mask_path):
                continue
            mask = read_tif(mask_path)[0]
            label = int((mask == 1).any())
            data.append((img.astype(np.float32), label))
    print("MADOS samples:", len(data))
    return data


def load_plp2019():
    """
    Загрузка PLP2019 с масштабированием маленьких изображений до TARGET_SIZE.
    """
    data = []
    if not os.path.exists(PLP2019_ROOT):
        print("PLP2019 root not found, skipping.")
        return data

    nc_folder = os.path.join(PLP2019_ROOT, "S2_satellite_images_nc")
    vector_root = os.path.join(PLP2019_ROOT, "Vector_Points")

    if not os.path.exists(nc_folder) or not os.path.exists(vector_root):
        print("PLP2019: missing required subfolders, skipping.")
        return data

    # Целевые длины волн для каналов Sentinel-2 в порядке: B2, B3, B4, B5, B6, B7, B8, B11, B12
    target_wavelengths = [492, 560, 665, 704, 740, 783, 833, 1614, 2202]

    nc_files = [f for f in os.listdir(nc_folder) if f.endswith('.nc')]
    date_pattern = re.compile(r'(\d{8})')

    for nc_file in tqdm(nc_files, desc="PLP2019"):
        match = date_pattern.search(nc_file)
        if not match:
            continue
        date_str = match.group(1)

        # Поиск соответствующей папки с шейп-файлами
        vector_date_folder = os.path.join(vector_root, date_str)
        if not os.path.exists(vector_date_folder):
            candidates = [d for d in os.listdir(vector_root) if date_str in d]
            if not candidates:
                continue
            vector_date_folder = os.path.join(vector_root, candidates[0])

        shp_files = [f for f in os.listdir(vector_date_folder) if f.endswith('.shp')]
        if not shp_files:
            continue
        shp_file = os.path.join(vector_date_folder, shp_files[0])

        nc_path = os.path.join(nc_folder, nc_file)
        try:
            img, lat, lon = read_nc(nc_path, target_wavelengths)
        except Exception as e:
            print(f"Error reading {nc_path}: {e}")
            continue

        img = np.nan_to_num(img)
        if img.max() > 10:
            img = img / 10000.0

        ind = indices(img)
        img = np.concatenate([img, ind], axis=0)  # (12, H, W)
        img = np.clip(img, *IMG_CLIP)

        H, W = img.shape[1], img.shape[2]

        # Определение геотрансформации
        transform = None
        if lat is not None and lon is not None:
            lat_min, lat_max = lat.min(), lat.max()
            lon_min, lon_max = lon.min(), lon.max()
            pixel_width = (lon_max - lon_min) / W
            pixel_height = (lat_max - lat_min) / H
            from rasterio.transform import from_origin
            transform = from_origin(lon_min, lat_max, pixel_width, pixel_height)
        else:
            transform = from_origin(0, H, 1, -1)  # fallback

        # Чтение шейп-файла и растеризация
        try:
            with fiona.open(shp_file, 'r') as shapefile:
                fields = shapefile.schema['properties']
                print(f"  Fields in {shp_file}: {list(fields.keys())}")  # отладка
                percent_field = None
                for field in fields:
                    if field.lower() in ['percent', 'pct', 'value', 'gridcode', 'percentage', 'PCT']:
                        percent_field = field
                        break
                if percent_field is None:
                    print(f"  No percent field found, using all points as plastic.")
                    # Если поле не найдено, используем все точки
                shapes = []
                for feature in shapefile:
                    geom = shape(feature['geometry'])
                    if percent_field is not None:
                        percent = feature['properties'].get(percent_field, 0)
                        if percent is None:
                            percent = 0
                        if isinstance(percent, str):
                            try:
                                percent = float(percent)
                            except:
                                percent = 0
                        if percent > 0:
                            shapes.append((geom, 1))
                    else:
                        # Если поле не указано, считаем любую точку пластиком
                        shapes.append((geom, 1))

                if not shapes:
                    print(f"  No shapes with plastic for {date_str}, skipping.")
                    continue

                # Растеризация на исходном размере
                mask = rasterize(
                    shapes,
                    out_shape=(H, W),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
                plastic_pixels = np.sum(mask > 0)
                print(f"  Rasterized mask: {plastic_pixels} plastic pixels")
        except Exception as e:
            print(f"Error processing shapefile for {date_str}: {e}")
            continue

        # Масштабируем изображение и маску до TARGET_SIZE
        if (H, W) != TARGET_SIZE:
            img = resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.float32), TARGET_SIZE[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            H, W = TARGET_SIZE

        # Нарезка патчей с перекрытием
        stride = TARGET_SIZE[0] // 2
        patch_count = 0
        for y in range(0, H - TARGET_SIZE[0] + 1, stride):
            for x in range(0, W - TARGET_SIZE[1] + 1, stride):
                patch = img[:, y:y+TARGET_SIZE[0], x:x+TARGET_SIZE[1]]
                mask_patch = mask[y:y+TARGET_SIZE[0], x:x+TARGET_SIZE[1]]
                if mask_patch.shape != (TARGET_SIZE[0], TARGET_SIZE[1]):
                    continue
                plastic_ratio = mask_patch.mean()
                label = 1 if plastic_ratio >= PLASTIC_THRESHOLD else 0
                data.append((patch.astype(np.float32), label))
                patch_count += 1
        print(f"  Generated {patch_count} patches for date {date_str}")

    print("PLP2019 samples:", len(data))
    return data


def save_splits(all_data, out_dir=OUT, seed=RSEED):
    X = np.array(all_data, dtype=object)
    labels = np.array([int(x[1]) for x in X])
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.10, stratify=labels, random_state=seed)
    labels_trainval = labels[idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15, stratify=labels_trainval, random_state=seed)
    np.save(os.path.join(out_dir, "train.npy"), X[idx_train])
    np.save(os.path.join(out_dir, "val.npy"), X[idx_val])
    np.save(os.path.join(out_dir, "test.npy"), X[idx_test])
    return idx_train, idx_val, idx_test


def compute_meta(train_array):
    C = EXPECTED_CHANNELS
    channel_sum = np.zeros(C, dtype=np.float64)
    channel_sq = np.zeros(C, dtype=np.float64)
    pixels = 0
    for img, _ in tqdm(train_array, desc="Computing meta"):
        flat = img.reshape(C, -1)
        channel_sum += flat.sum(1)
        channel_sq += (flat ** 2).sum(1)
        pixels += flat.shape[1]
    mean = channel_sum / pixels
    std = np.sqrt(channel_sq / pixels - mean ** 2)
    std = np.clip(std, 1e-6, None)
    return mean.tolist(), std.tolist()


def main():
    marida = load_marida()
    mados = load_mados()
    plp2019 = load_plp2019()

    all_data = marida + mados + plp2019
    print("Total samples:", len(all_data))
    if len(all_data) == 0:
        raise RuntimeError("No samples found. Check data folders.")

    idx_train, idx_val, idx_test = save_splits(all_data)

    train_arr = np.load(os.path.join(OUT, "train.npy"), allow_pickle=True)
    mean, std = compute_meta(train_arr)
    meta = {
        "channels": EXPECTED_CHANNELS,
        "mean": mean,
        "std": std
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved train/val/test and meta.json in", OUT)


if __name__ == "__main__":
    main()