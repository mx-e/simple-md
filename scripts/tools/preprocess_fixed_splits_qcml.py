#! /usr/bin/python3

import tensorflow_datasets as tfds
import tensorflow as tf
#prevent tensorflow from allocating all memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import os
import json
import numpy as np
from typing import Dict, List, Tuple
import random
import hashlib
import resource


def increase_file_limit():
    """
    Attempt to increase the file descriptor limit
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current file limits: soft={soft}, hard={hard}")

        # Try to increase to hard limit
        desired_limit = min(hard, 65536)  # Set to reasonable maximum
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard))

        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"New file limits: soft={new_soft}, hard={new_hard}")
    except Exception as e:
        print(f"Warning: Could not increase file limit: {e}")


def get_split_hash(key, seed):
    """
    TensorFlow compatible hashing function
    """
    hash_input = tf.strings.join([key, tf.constant(str(seed))], separator="_")
    hash_value = tf.strings.to_hash_bucket_fast(hash_input, num_buckets=1_000_000)
    return tf.cast(hash_value, tf.float32) / 1_000_000.0


def determine_split(example, split_by_smiles, train_ratio, val_ratio, seed):
    """
    Uses TensorFlow operations to determine the split
    """
    key = example["smiles"] if split_by_smiles else example["key_hash"]
    hash_value = get_split_hash(key, seed)

    return tf.case(
        [
            (hash_value < train_ratio, lambda: tf.constant("train")),
            (hash_value < train_ratio + val_ratio, lambda: tf.constant("val")),
        ],
        default=lambda: tf.constant("test"),
    )


def create_streaming_splits(
    dataset: tf.data.Dataset,
    split_by_smiles: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, tf.data.Dataset]:
    """
    Creates train/val/test splits while maintaining streaming behavior.
    """

    def create_filter(split_name: str):
        return (
            lambda x: determine_split(x, split_by_smiles, train_ratio, val_ratio, seed)
            == split_name
        )

    split_datasets = {}
    for split_name in ["train", "val", "test"]:
        split_datasets[split_name] = dataset.filter(create_filter(split_name))

    return split_datasets


def merge_and_save_as_tfrecord(
    data_dir: str,
    fast_fs_dir: str,
    collate_ds_dirs: List[str],
    ds_out_name: str,
    ds_in_name: str,    
    filter_outliers: bool = True,
    test_results: bool = False,
    limit_samples: int = None,
    split_by_smiles: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Merges and saves the dataset with streaming splits.
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Split ratios must sum to 1"

    # cp to fast fs dir
    print("Copying to fast fs dir")
    os.makedirs(fast_fs_dir, exist_ok=True)
    os.system(f"cp -r {data_dir}/{ds_in_name} {fast_fs_dir}/")

    # increasing open file limit
    increase_file_limit()

    print("Loading datasets")
    read_config = tfds.ReadConfig(interleave_cycle_length=1)
    datasets = [
        tfds.load(
            f"{ds_in_name}/{dir_name}",
            split="full",
            data_dir=FAST_FS_DIR,
            read_config=read_config,
        )
        for dir_name in collate_ds_dirs
    ]

    def merge_features(*args):
        merged_map = {}
        for ds in args:
            for k, v in ds.items():
                merged_map[k] = v
        return merged_map

    def filter_outliers_func(example):
        return not example["is_outlier"] if filter_outliers else True

    # Create the base dataset with merged features and outlier filtering
    print("Merging and filtering dataset")
    zipped_ds = (
        tf.data.Dataset.zip(tuple(datasets))
        .map(merge_features)
        .filter(filter_outliers_func)
        .prefetch(10000)
    )
    if limit_samples:
        zipped_ds = zipped_ds.take(limit_samples)

    # Create streaming splits
    print("Creating splits")
    split_datasets = create_streaming_splits(
        zipped_ds,
        split_by_smiles=split_by_smiles,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    print(zipped_ds.take(1))

    # Define the features dictionary
    features_dict = {
        "is_outlier": tfds.features.Scalar(dtype=np.bool_),
        "atomic_numbers": tfds.features.Tensor(shape=(None,), dtype=np.uint8),
        "key_hash": tfds.core.features.Text(),
        "molecular_weight": tfds.features.Tensor(shape=(), dtype=np.float64),
        "chemical_formula": tfds.features.Text(),
        "multiplicity": tfds.features.Scalar(dtype=np.int64),
        "conformation_seq": tfds.features.Scalar(dtype=np.int64),
        "charge": tfds.features.Scalar(dtype=np.int64),
        "conformation_parent_seq": tfds.features.Scalar(dtype=np.int64),
        "smiles": tfds.features.Text(),
        "smiles_hash": tfds.features.Text(),
        "num_heavy_atoms": tfds.features.Scalar(dtype=np.int64),
        "num_atoms": tfds.features.Scalar(dtype=np.int64),
        "pbe0_electronic_free_energy": tfds.features.Tensor(shape=(), dtype=np.float64),
        "pbe0_energy": tfds.features.Tensor(shape=(), dtype=np.float64),
        "pbe0_forces": tfds.features.Tensor(shape=(None, 3), dtype=np.float32),
        "pbe0_formation_energy": tfds.features.Tensor(shape=(), dtype=np.float64),
        "pbe0_dipole": tfds.features.Tensor(shape=(3,), dtype=np.float32),
        "positions": tfds.features.Tensor(shape=(None, 3), dtype=np.float32),
    }

    # Store as TFDS dataset
    print("Storing as TFDS dataset...")
    tfds.dataset_builders.store_as_tfds_dataset(
        name=ds_out_name,
        version="1.0.0",
        data_dir=fast_fs_dir,
        split_datasets=split_datasets,
        features=tfds.features.FeaturesDict(features_dict),
        description="Unified QCML Public Dataset with filtered outliers and streaming train/val/test splits",
        release_notes={
            "1.0.0": "Initial release with streaming train/val/test splits",
        },
    )

    print("Copying back to original dir")
    output_dir = os.path.join(data_dir, ds_out_name)
    os.makedirs(output_dir, exist_ok=True)
    fast_out_dir = os.path.join(fast_fs_dir, ds_out_name)
    os.system(f"cp -r {fast_out_dir}/* {output_dir}/")
    print(f"Dataset saved to: {output_dir}")
    if test_results:
        test_splits(ds_out_name, data_dir=data_dir)


def test_splits(ds_name, data_dir="."):
    from collections import defaultdict  # dynamic import for testing

    """
    Comprehensive test of the dataset splits.
    """
    print("Loading all splits...")
    splits = {
        "train": tfds.load(ds_name, split="train", data_dir=data_dir),
        "val": tfds.load(ds_name, split="val", data_dir=data_dir),
        "test": tfds.load(ds_name, split="test", data_dir=data_dir),
    }

    # Convert to lists for analysis
    split_data = {split_name: list(ds) for split_name, ds in splits.items()}

    # 1. Basic split size analysis
    print("\n1. Split Sizes:")
    total_samples = sum(len(data) for data in split_data.values())
    for split_name, data in split_data.items():
        split_ratio = len(data) / total_samples
        print(f"{split_name}: {len(data)} samples ({split_ratio:.3%})")

    # 2. Check SMILES distribution
    print("\n2. SMILES Analysis:")
    smiles_to_splits = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for split_name, data in split_data.items():
        for example in data:
            smiles = example["smiles"].numpy().decode("utf-8")
            smiles_to_splits[smiles][split_name] += 1

    # Check for SMILES appearing in multiple splits
    violations = 0
    for smiles, split_counts in smiles_to_splits.items():
        splits_present = sum(1 for count in split_counts.values() if count > 0)
        if splits_present > 1:
            violations += 1
            if violations <= 5:  # Show first 5 violations
                print(f"SMILES '{smiles}' appears in multiple splits: {split_counts}")

    total_unique_smiles = len(smiles_to_splits)
    print(f"\nTotal unique SMILES: {total_unique_smiles}")
    print(f"SMILES appearing in multiple splits: {violations}")
    if violations == 0:
        print("✓ All SMILES are correctly confined to single splits")

    # 3. Conformer analysis
    print("\n3. Conformer Analysis:")
    conformers_per_smiles = defaultdict(int)
    for split_name, data in split_data.items():
        for example in data:
            smiles = example["smiles"].numpy().decode("utf-8")
            conformers_per_smiles[smiles] += 1

    conformer_counts = list(conformers_per_smiles.values())
    print(f"Average conformers per SMILES: {np.mean(conformer_counts):.2f}")
    print(f"Median conformers per SMILES: {np.median(conformer_counts):.2f}")
    print(f"Min conformers: {min(conformer_counts)}")
    print(f"Max conformers: {max(conformer_counts)}")

    # 4. Property distributions across splits
    print("\n4. Property Distributions:")
    properties = ["molecular_weight", "num_atoms", "num_heavy_atoms"]

    for prop in properties:
        print(f"\n{prop}:")
        for split_name, data in split_data.items():
            values = [float(example[prop]) for example in data]
            print(f"{split_name}:")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Std:  {np.std(values):.2f}")
            print(f"  Min:  {np.min(values):.2f}")
            print(f"  Max:  {np.max(values):.2f}")

    # 5. Check energy ranges and outliers
    print("\n5. Energy Analysis:")
    for split_name, data in split_data.items():
        energies = [float(example["pbe0_energy"]) for example in data]
        print(f"\n{split_name} energy statistics:")
        print(f"  Mean: {np.mean(energies):.2f}")
        print(f"  Std:  {np.std(energies):.2f}")
        print(f"  Min:  {np.min(energies):.2f}")
        print(f"  Max:  {np.max(energies):.2f}")

        # Check for potential outliers (more than 3 std from mean)
        mean, std = np.mean(energies), np.std(energies)
        outliers = sum(1 for e in energies if abs(e - mean) > 3 * std)
        print(f"  Potential outliers (>3σ): {outliers} ({outliers/len(energies):.2%})")

    # 6. give first example in each split
    for split in ["train", "val", "test"]:
        loaded_ds = tfds.load(ds_name, split=split, data_dir=data_dir)
        print(f"\nFirst example in the {split} split:")
        for example in loaded_ds.take(1):
            print(example)


if __name__ == "__main__":
    DATA_DIR = "."
    FAST_FS_DIR = "/temp/mx-ds"
    COLLATE_DS_DIRS = [
        "qm7x_atomic_numbers",
        "qm7x_is_outlier",
        "qm7x_metadata",
        "qm7x_pbe0_electronic_free_energy",
        "qm7x_pbe0_energy",
        "qm7x_pbe0_forces",
        "qm7x_pbe0_dipole",
        "qm7x_pbe0_formation_energy",
        "qm7x_positions",
    ]
    DS_IN_NAME = "qm7x_pbe0"
    DS_OUT_NAME = "qm7x_pbe0_split_by_smiles"
    TEST_RUN = False
    SPLIT_BY_SMILES = True
    SEED = 42
    merge_and_save_as_tfrecord(
        data_dir=DATA_DIR,
        fast_fs_dir=FAST_FS_DIR,
        collate_ds_dirs=COLLATE_DS_DIRS,
        ds_in_name=DS_IN_NAME,
        ds_out_name=DS_OUT_NAME,
        filter_outliers=True,
        limit_samples=250000 if TEST_RUN else None,
        test_results=TEST_RUN,  # Set to True to run the test after saving (this might be memory intensive for large datasets)
        split_by_smiles=SPLIT_BY_SMILES,  # Set to True to split by unique molecules, False to split by sample hash
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=SEED
    )
