"""
Generate train.pkl / test.pkl / val.pkl from raw CUB-200-2011 data files.

These pkl files are required by tti.py and probe.py for TTI evaluation
and linear probe experiments. The main training pipeline (DatasetBirds)
does NOT use these files — it reads raw data directly via ImageFolder.

Usage:
    python src/data/generate_cub_pkl.py [--cub_dir PATH] [--output_dir PATH]

Default paths come from .env / config.py (CUB_DATA_DIR).
Output goes to <output_dir>/class_attr_data_10/ by default, matching
tti.py's default data_dir2='class_attr_data_10'.

Each pkl entry contains:
    img_path            : str   (relative path under CUB dir, e.g. 'images/001.Black_footed_Albatross/...')
    class_label         : int   (0–199)
    attribute_label     : list  (312 binary labels, 0 or 1)
    attribute_certainty : list  (312 integer certainty ids: 1=not visible, 2=guessing, 3=probably, 4=definitely)
    uncertain_attribute_label : list (312 floats, weighted by certainty)
"""
import os
import sys
import pickle
import argparse

import numpy as np

# Try loading project config; fall back to defaults if unavailable
try:
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from util.config import CUB_DATA_DIR as _CUB_DATA_DIR, N_ATTRIBUTES as _N_ATTRIBUTES, N_CLASSES as _N_CLASSES
    DEFAULT_CUB_DIR = _CUB_DATA_DIR
    DEFAULT_N_ATTR = _N_ATTRIBUTES
    DEFAULT_N_CLASSES = _N_CLASSES
except ImportError:
    DEFAULT_CUB_DIR = 'src/data/CUB_200_2011'
    DEFAULT_N_ATTR = 312
    DEFAULT_N_CLASSES = 200


def load_image_splits(cub_dir):
    """Return dict {image_id: is_train} from train_test_split.txt."""
    splits = {}
    with open(os.path.join(cub_dir, 'train_test_split.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                splits[int(parts[0])] = int(parts[1])
    return splits


def load_image_paths(cub_dir):
    """Return dict {image_id: relative_path} from images.txt."""
    paths = {}
    with open(os.path.join(cub_dir, 'images.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) >= 2:
                paths[int(parts[0])] = parts[1]
    return paths


def load_class_labels(cub_dir):
    """Return dict {image_id: class_label (0-indexed)} from image_class_labels.txt."""
    labels = {}
    with open(os.path.join(cub_dir, 'image_class_labels.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[int(parts[0])] = int(parts[1]) - 1  # 0-indexed
    return labels


def load_image_attributes(cub_dir):
    """Return dict {image_id: {attr_id: (is_present, certainty_id)}} from image_attribute_labels.txt."""
    attrs = {}
    filepath = os.path.join(cub_dir, 'attributes/image_attribute_labels.txt')
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            img_id = int(parts[0])
            attr_id = int(parts[1])
            is_present = int(parts[2])
            certainty_id = int(parts[3])
            # certainty_value = float(parts[4])  # not needed
            if img_id not in attrs:
                attrs[img_id] = {}
            attrs[img_id][attr_id] = (is_present, certainty_id)
    return attrs


def build_pkl_entries(cub_dir, split_filter, image_splits, image_paths,
                      class_labels, image_attrs, n_attributes=DEFAULT_N_ATTR):
    """Build list of pkl dict entries for images matching split_filter."""
    entries = []
    for img_id in sorted(image_splits.keys()):
        is_train = image_splits[img_id]
        if is_train != split_filter:
            continue

        img_path = os.path.join(cub_dir, 'images', image_paths[img_id])
        class_label = class_labels[img_id]

        attr_dict = image_attrs.get(img_id, {})
        attribute_label = []
        attribute_certainty = []
        uncertain_attribute_label = []

        for attr_idx in range(1, n_attributes + 1):
            is_present, cert_id = attr_dict.get(attr_idx, (0, 1))
            attribute_label.append(is_present)
            attribute_certainty.append(cert_id)
            # Weight by certainty: 1=not visible(0), 2=guessing(0.5), 3=probably(0.75), 4=definitely(1.0)
            cert_weights = {1: 0.0, 2: 0.5, 3: 0.75, 4: 1.0}
            uncertain_attribute_label.append(
                is_present * cert_weights.get(cert_id, 1.0)
            )

        entries.append({
            'img_path': img_path,
            'class_label': class_label,
            'attribute_label': attribute_label,
            'attribute_certainty': attribute_certainty,
            'uncertain_attribute_label': uncertain_attribute_label,
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description='Generate CUB pkl files for TTI/Probe evaluation')
    parser.add_argument('--cub_dir', default=DEFAULT_CUB_DIR,
                        help='Root directory of CUB-200-2011 dataset')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for pkl files (default: <cub_dir>/class_attr_data_10)')
    args = parser.parse_args()

    cub_dir = args.cub_dir
    if args.output_dir is None:
        output_dir = os.path.join(cub_dir, 'class_attr_data_10')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f'CUB dir: {cub_dir}')
    print(f'Output dir: {output_dir}')

    # Load raw data
    image_splits = load_image_splits(cub_dir)
    image_paths = load_image_paths(cub_dir)
    class_labels = load_class_labels(cub_dir)
    image_attrs = load_image_attributes(cub_dir)

    n_images = len(image_splits)
    print(f'Total images: {n_images}')
    print(f'Images with attribute data: {len(image_attrs)}')

    # Build train / val / test splits
    # CUB train_test_split: 1 = train, 0 = test
    # For val.pkl, we use the first half of the test set (standard CBM convention)
    train_entries = build_pkl_entries(cub_dir, 1, image_splits, image_paths,
                                       class_labels, image_attrs)
    test_entries_all = build_pkl_entries(cub_dir, 0, image_splits, image_paths,
                                          class_labels, image_attrs)

    # Split test into val + test (50/50, matching original CBM convention)
    n_test = len(test_entries_all)
    val_entries = test_entries_all[:n_test // 2]
    test_entries = test_entries_all[n_test // 2:]

    print(f'Train entries: {len(train_entries)}')
    print(f'Val entries: {len(val_entries)}')
    print(f'Test entries: {len(test_entries)}')

    # Save pkl files
    for name, data in [('train', train_entries), ('val', val_entries), ('test', test_entries)]:
        pkl_path = os.path.join(output_dir, f'{name}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved {pkl_path} ({len(data)} entries)')

    print('Done!')


if __name__ == '__main__':
    main()