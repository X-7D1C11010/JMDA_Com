"""
Prepare image list files for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光 (Backlight), 雨天 (Rainy), 雾天 (Foggy), 黑天 (Dark)
Modality: 可见光 (Visible light)
Classes: 1-14 (0-indexed: 0-13)
"""
import os

DATA_ROOT = r'd:\Code\JMDA-Net\Data'
OUTPUT_DIR = r'd:\Code\JMDA_Com\data_lists'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAINS = {
    'sunny': '晴天',
    'backlight': '逆光',
    'rainy': '雨天',
    'foggy': '雾天',
    'dark': '黑天',
}

CLASS_NAMES = [f'class_{i}' for i in range(1, 15)]


def collect_images(domain_cn, split, modality='可见光'):
    """Collect (image_path, label) pairs for a domain/split/modality."""
    split_path = os.path.join(DATA_ROOT, domain_cn, split)
    if not os.path.exists(split_path):
        return []

    items = os.listdir(split_path)
    pairs = []

    if modality in items:
        # Sunny structure: modality -> class
        mod_path = os.path.join(split_path, modality)
        for cls_name in sorted(os.listdir(mod_path), key=lambda x: int(x)):
            cls_path = os.path.join(mod_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            label = int(cls_name) - 1  # 0-indexed
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    pairs.append((os.path.join(cls_path, fname), label))
    else:
        # Other structure: class -> modality
        for cls_name in sorted(items, key=lambda x: int(x)):
            cls_path = os.path.join(split_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            mod_path = os.path.join(cls_path, modality)
            if not os.path.exists(mod_path):
                continue
            label = int(cls_name) - 1
            for fname in os.listdir(mod_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    pairs.append((os.path.join(mod_path, fname), label))

    return pairs


def write_list(pairs, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for path, label in pairs:
            f.write(f'{path} {label}\n')
    return len(pairs)


def main():
    for domain_en, domain_cn in DOMAINS.items():
        for split in ['train', 'val']:
            pairs = collect_images(domain_cn, split, '可见光')
            out_path = os.path.join(OUTPUT_DIR, f'{domain_en}_{split}.txt')
            n = write_list(pairs, out_path)
            print(f'{domain_en}_{split}: {n} images -> {out_path}')

    # Print class distribution for source domain
    print('\n--- Source (sunny) class distribution (train, visible) ---')
    src_pairs = collect_images('晴天', 'train', '可见光')
    from collections import Counter
    cnt = Counter(l for _, l in src_pairs)
    for cls in sorted(cnt):
        print(f'  class {cls+1}: {cnt[cls]}')


if __name__ == '__main__':
    main()
