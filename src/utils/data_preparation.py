"""
Data preparation utilities for converting client data format to optimizer format.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import math


def prepare_client_data_for_optimizer(
    embeddings_by_shape: List[np.ndarray],
    metadata_by_shape: List[pd.DataFrame],
    group_names: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert per-shape client data to unified optimizer format.

    The client provides separate arrays and DataFrames for each shape. This function
    merges them into a single embeddings array and unified metadata DataFrame with
    distribution_id, shape logits, and all shape parameters.

    Args:
        embeddings_by_shape: List of (n_samples, 400) arrays, one per shape
        metadata_by_shape: List of DataFrames with distribution_id and shape-specific params
        group_names: Shape names matching the order of input lists
                     (e.g., ['circle', 'ellipse', 'irregular'])

    Returns:
        synthetic_embeddings: (total_samples, 400) concatenated array
        metadata_df: Unified DataFrame with columns:
                     - distribution_id: int
                     - shape_logit_{shape}: float for each shape
                     - {shape}__{param}: float for each shape parameter

    Example:
        >>> circle_embs = np.random.randn(100, 400)
        >>> ellipse_embs = np.random.randn(80, 400)
        >>> irregular_embs = np.random.randn(70, 400)
        >>>
        >>> circle_df = pd.DataFrame({
        ...     'distribution_id': [0]*50 + [1]*50,
        ...     'void_count_mean': [5.0]*100,
        ...     'base_size_std': [2.0]*100
        ... })
        >>> # ... similar for ellipse_df, irregular_df
        >>>
        >>> embeddings, metadata = prepare_client_data_for_optimizer(
        ...     [circle_embs, ellipse_embs, irregular_embs],
        ...     [circle_df, ellipse_df, irregular_df],
        ...     ['circle', 'ellipse', 'irregular']
        ... )
    """
    if len(embeddings_by_shape) != len(metadata_by_shape):
        raise ValueError(
            f"Mismatch: {len(embeddings_by_shape)} embedding arrays but "
            f"{len(metadata_by_shape)} metadata DataFrames"
        )

    if len(embeddings_by_shape) != len(group_names):
        raise ValueError(
            f"Mismatch: {len(embeddings_by_shape)} data sources but "
            f"{len(group_names)} group names"
        )

    # Step 1: Count samples per distribution per shape
    sample_counts = _count_samples_per_distribution(metadata_by_shape, group_names)

    # Step 2: Compute shape logits for each distribution
    logits_by_dist = _compute_shape_logits(sample_counts, group_names)

    # Step 3: Merge parameters from all shape DataFrames
    params_by_dist = _merge_shape_parameters(metadata_by_shape, group_names)

    # Step 4: Build unified metadata rows (one per sample)
    unified_metadata_rows = []

    for shape_idx, (shape_name, metadata_df) in enumerate(zip(group_names, metadata_by_shape)):
        for _, row in metadata_df.iterrows():
            dist_id = int(row['distribution_id'])

            # Build row: dist_id + logits + all shape params
            unified_row = {
                'distribution_id': dist_id,
                **logits_by_dist[dist_id],
                **params_by_dist[dist_id]
            }

            unified_metadata_rows.append(unified_row)

    metadata_df_unified = pd.DataFrame(unified_metadata_rows)

    # Step 5: Concatenate embeddings in same order as metadata
    synthetic_embeddings = np.concatenate(embeddings_by_shape, axis=0)

    # Verify shapes match
    if len(synthetic_embeddings) != len(metadata_df_unified):
        raise RuntimeError(
            f"Internal error: {len(synthetic_embeddings)} embeddings but "
            f"{len(metadata_df_unified)} metadata rows"
        )

    return synthetic_embeddings, metadata_df_unified


def _count_samples_per_distribution(
    metadata_by_shape: List[pd.DataFrame],
    group_names: List[str]
) -> Dict[int, Dict[str, int]]:
    """
    Count how many samples each distribution has for each shape.

    Returns:
        {dist_id: {shape_name: count}}
        Example: {0: {'circle': 12, 'ellipse': 8, 'irregular': 5}}
    """
    # Get all unique distribution IDs across all shapes
    all_dist_ids = set()
    for df in metadata_by_shape:
        all_dist_ids.update(df['distribution_id'].unique())

    sample_counts = {}

    for dist_id in sorted(all_dist_ids):
        sample_counts[dist_id] = {}

        for shape_name, metadata_df in zip(group_names, metadata_by_shape):
            # Count rows with this distribution_id
            count = len(metadata_df[metadata_df['distribution_id'] == dist_id])
            sample_counts[dist_id][shape_name] = count

    return sample_counts


def _compute_shape_logits(
    sample_counts: Dict[int, Dict[str, int]],
    group_names: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    Compute shape logits from sample counts using inverse softmax.

    Args:
        sample_counts: {dist_id: {shape: count}}
        group_names: List of shape names

    Returns:
        {dist_id: {shape_logit_{shape}: logit_value}}
    """
    logits_by_dist = {}

    for dist_id, counts in sample_counts.items():
        # Total samples for this distribution
        total = sum(counts.values())

        if total == 0:
            raise ValueError(f"Distribution {dist_id} has 0 total samples")

        # Compute probabilities
        probs = {shape: counts.get(shape, 0) / total for shape in group_names}

        # Inverse softmax: logit = log(prob)
        # Clamp to avoid log(0)
        logits = {}
        for shape in group_names:
            prob = max(probs[shape], 1e-6)  # Avoid log(0)
            logits[f'shape_logit_{shape}'] = math.log(prob)

        logits_by_dist[dist_id] = logits

    return logits_by_dist


def _merge_shape_parameters(
    metadata_by_shape: List[pd.DataFrame],
    group_names: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    Merge shape-specific parameters into unified format with prefixes.

    Args:
        metadata_by_shape: List of DataFrames with shape-specific params
        group_names: List of shape names

    Returns:
        {dist_id: {shape__param: value}}
        Example: {0: {'circle__void_count_mean': 5.0, 'ellipse__rotation_std': 15.0, ...}}
    """
    # Get all unique distribution IDs
    all_dist_ids = set()
    for df in metadata_by_shape:
        all_dist_ids.update(df['distribution_id'].unique())

    params_by_dist = {}

    for dist_id in sorted(all_dist_ids):
        merged_params = {}

        for shape_name, metadata_df in zip(group_names, metadata_by_shape):
            # Get rows for this distribution
            shape_rows = metadata_df[metadata_df['distribution_id'] == dist_id]

            if len(shape_rows) == 0:
                # No samples for this shape in this distribution
                # This is allowed (shape has 0 probability)
                continue

            # Take first row (all rows within same dist_id should have identical params)
            row = shape_rows.iloc[0]

            # Add all columns except distribution_id with shape prefix
            for col in metadata_df.columns:
                if col != 'distribution_id':
                    merged_params[f'{shape_name}__{col}'] = float(row[col])

        params_by_dist[dist_id] = merged_params

    return params_by_dist
