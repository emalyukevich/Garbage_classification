import os
import shutil
import random
from tqdm import tqdm

# Задаем пропорции
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Получаем базовую директорию (на уровень выше notebooks/)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Пути к директориям
raw_data_dir = os.path.join(base_dir, 'data', 'raw', 'Garbage classification')
train_dir = os.path.join(base_dir, 'data', 'train')
val_dir = os.path.join(base_dir, 'data', 'val')
test_dir = os.path.join(base_dir, 'data', 'test')

# Удаляем старые директории, если они есть
for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Обработка по классам
for class_name in os.listdir(raw_data_dir):
    class_path = os.path.join(raw_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_count = int(train_ratio * total)
    val_count = int(val_ratio * total)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    # Копируем в соответствующие папки
    for subset, img_list in zip([train_dir, val_dir, test_dir], [train_imgs, val_imgs, test_imgs]):
        class_subset_dir = os.path.join(subset, class_name)
        os.makedirs(class_subset_dir, exist_ok=True)
        for img in tqdm(img_list, desc=f"{subset.split(os.sep)[-1]} - {class_name}"):
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(class_subset_dir, img)
            shutil.copy2(src_path, dst_path)


