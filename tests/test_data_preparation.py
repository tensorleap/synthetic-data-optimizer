"""
Tests for data preparation utilities.
"""

import numpy as np
import pandas as pd
import pytest
import math

from src.utils.data_preparation import prepare_client_data_for_optimizer


def test_prepare_client_data_basic():
    """Test basic conversion with 2 distributions and 3 shapes."""
    # Distribution 0: 10 circle, 5 ellipse, 3 irregular = 18 total
    # Distribution 1: 8 circle, 12 ellipse, 6 irregular = 26 total

    # Create mock embeddings
    circle_embs = np.random.randn(18, 400).astype(np.float32)  # 10 + 8
    ellipse_embs = np.random.randn(17, 400).astype(np.float32)  # 5 + 12
    irregular_embs = np.random.randn(9, 400).astype(np.float32)  # 3 + 6

    # Create mock metadata
    circle_df = pd.DataFrame({
        'distribution_id': [0]*10 + [1]*8,
        'void_count_mean': [5.2]*10 + [6.1]*8,
        'base_size_std': [2.1]*10 + [2.4]*8
    })

    ellipse_df = pd.DataFrame({
        'distribution_id': [0]*5 + [1]*12,
        'void_count_mean': [4.3]*5 + [5.2]*12,
        'rotation_mean': [45.5]*5 + [90.8]*12
    })

    irregular_df = pd.DataFrame({
        'distribution_id': [0]*3 + [1]*6,
        'void_count_mean': [3.4]*3 + [4.3]*6,
        'complexity_mean': [2.5]*3 + [3.1]*6
    })

    # Convert
    embeddings, metadata_df = prepare_client_data_for_optimizer(
        [circle_embs, ellipse_embs, irregular_embs],
        [circle_df, ellipse_df, irregular_df],
        ['circle', 'ellipse', 'irregular']
    )

    # Verify embeddings shape
    assert embeddings.shape == (44, 400)  # 18 + 17 + 9

    # Verify metadata shape
    assert len(metadata_df) == 44
    assert 'distribution_id' in metadata_df.columns

    # Verify logit columns exist
    assert 'shape_logit_circle' in metadata_df.columns
    assert 'shape_logit_ellipse' in metadata_df.columns
    assert 'shape_logit_irregular' in metadata_df.columns

    # Verify param columns exist with prefixes
    assert 'circle__void_count_mean' in metadata_df.columns
    assert 'circle__base_size_std' in metadata_df.columns
    assert 'ellipse__void_count_mean' in metadata_df.columns
    assert 'ellipse__rotation_mean' in metadata_df.columns
    assert 'irregular__void_count_mean' in metadata_df.columns
    assert 'irregular__complexity_mean' in metadata_df.columns

    # Verify distribution IDs
    assert set(metadata_df['distribution_id'].unique()) == {0, 1}

    # Verify row counts per distribution
    dist_0_rows = metadata_df[metadata_df['distribution_id'] == 0]
    dist_1_rows = metadata_df[metadata_df['distribution_id'] == 1]
    assert len(dist_0_rows) == 18  # 10 + 5 + 3
    assert len(dist_1_rows) == 26  # 8 + 12 + 6


def test_logits_to_probabilities():
    """Test that logits convert back to correct probabilities via softmax."""
    # Simple case: 10 circle, 5 ellipse, 5 irregular
    circle_embs = np.random.randn(10, 400).astype(np.float32)
    ellipse_embs = np.random.randn(5, 400).astype(np.float32)
    irregular_embs = np.random.randn(5, 400).astype(np.float32)

    circle_df = pd.DataFrame({
        'distribution_id': [0]*10,
        'param1': [1.0]*10
    })
    ellipse_df = pd.DataFrame({
        'distribution_id': [0]*5,
        'param2': [2.0]*5
    })
    irregular_df = pd.DataFrame({
        'distribution_id': [0]*5,
        'param3': [3.0]*5
    })

    embeddings, metadata_df = prepare_client_data_for_optimizer(
        [circle_embs, ellipse_embs, irregular_embs],
        [circle_df, ellipse_df, irregular_df],
        ['circle', 'ellipse', 'irregular']
    )

    # Get logits from first row (all rows have same logits for same dist_id)
    first_row = metadata_df.iloc[0]
    logits = [
        first_row['shape_logit_circle'],
        first_row['shape_logit_ellipse'],
        first_row['shape_logit_irregular']
    ]

    # Convert to probabilities via softmax
    exp_logits = [math.exp(l) for l in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    # Expected probabilities: 10/20, 5/20, 5/20 = 0.5, 0.25, 0.25
    assert abs(probs[0] - 0.5) < 1e-6
    assert abs(probs[1] - 0.25) < 1e-6
    assert abs(probs[2] - 0.25) < 1e-6

    # Probabilities sum to 1
    assert abs(sum(probs) - 1.0) < 1e-6


def test_zero_samples_for_shape():
    """Test edge case where one shape has 0 samples for a distribution."""
    # Distribution 0: 10 circle, 0 ellipse, 5 irregular
    circle_embs = np.random.randn(10, 400).astype(np.float32)
    ellipse_embs = np.random.randn(0, 400).astype(np.float32)  # Empty
    irregular_embs = np.random.randn(5, 400).astype(np.float32)

    circle_df = pd.DataFrame({
        'distribution_id': [0]*10,
        'param1': [1.0]*10
    })
    ellipse_df = pd.DataFrame({
        'distribution_id': pd.Series([], dtype=int),  # Empty
        'param2': pd.Series([], dtype=float)
    })
    irregular_df = pd.DataFrame({
        'distribution_id': [0]*5,
        'param3': [3.0]*5
    })

    embeddings, metadata_df = prepare_client_data_for_optimizer(
        [circle_embs, ellipse_embs, irregular_embs],
        [circle_df, ellipse_df, irregular_df],
        ['circle', 'ellipse', 'irregular']
    )

    # Verify shape
    assert embeddings.shape == (15, 400)  # 10 + 0 + 5
    assert len(metadata_df) == 15

    # Verify ellipse logit is very low (0 samples → log(1e-6) ≈ -13.8)
    first_row = metadata_df.iloc[0]
    assert first_row['shape_logit_ellipse'] < -10  # Very negative


def test_parameter_values_preserved():
    """Test that parameter values from input DataFrames are correctly preserved."""
    circle_embs = np.random.randn(5, 400).astype(np.float32)
    ellipse_embs = np.random.randn(5, 400).astype(np.float32)

    circle_df = pd.DataFrame({
        'distribution_id': [0]*5,
        'void_count_mean': [5.2]*5,
        'base_size_std': [2.1]*5
    })
    ellipse_df = pd.DataFrame({
        'distribution_id': [0]*5,
        'void_count_mean': [4.3]*5,
        'rotation_mean': [45.5]*5
    })

    embeddings, metadata_df = prepare_client_data_for_optimizer(
        [circle_embs, ellipse_embs],
        [circle_df, ellipse_df],
        ['circle', 'ellipse']
    )

    # Check that all rows have correct param values
    for _, row in metadata_df.iterrows():
        assert row['circle__void_count_mean'] == 5.2
        assert row['circle__base_size_std'] == 2.1
        assert row['ellipse__void_count_mean'] == 4.3
        assert row['ellipse__rotation_mean'] == 45.5


def test_mismatched_inputs():
    """Test error handling for mismatched inputs."""
    circle_embs = np.random.randn(5, 400).astype(np.float32)
    circle_df = pd.DataFrame({'distribution_id': [0]*5, 'param1': [1.0]*5})

    # Mismatch: 1 embedding array but 2 DataFrames
    with pytest.raises(ValueError, match="Mismatch"):
        prepare_client_data_for_optimizer(
            [circle_embs],
            [circle_df, circle_df],
            ['circle']
        )

    # Mismatch: 1 embedding array but 2 group names
    with pytest.raises(ValueError, match="Mismatch"):
        prepare_client_data_for_optimizer(
            [circle_embs],
            [circle_df],
            ['circle', 'ellipse']
        )


def test_multiple_distributions():
    """Test with multiple distributions to ensure each gets correct logits and params."""
    # 3 distributions with different sample counts
    circle_embs = np.random.randn(30, 400).astype(np.float32)  # 10 + 10 + 10
    ellipse_embs = np.random.randn(27, 400).astype(np.float32)  # 5 + 12 + 10

    circle_df = pd.DataFrame({
        'distribution_id': [0]*10 + [1]*10 + [2]*10,
        'param1': [1.0]*10 + [2.0]*10 + [3.0]*10
    })
    ellipse_df = pd.DataFrame({
        'distribution_id': [0]*5 + [1]*12 + [2]*10,
        'param2': [10.0]*5 + [20.0]*12 + [30.0]*10
    })

    embeddings, metadata_df = prepare_client_data_for_optimizer(
        [circle_embs, ellipse_embs],
        [circle_df, ellipse_df],
        ['circle', 'ellipse']
    )

    # Verify each distribution has correct param values
    dist_0_rows = metadata_df[metadata_df['distribution_id'] == 0]
    dist_1_rows = metadata_df[metadata_df['distribution_id'] == 1]
    dist_2_rows = metadata_df[metadata_df['distribution_id'] == 2]

    # Distribution 0: circle param = 1.0, ellipse param = 10.0
    assert all(dist_0_rows['circle__param1'] == 1.0)
    assert all(dist_0_rows['ellipse__param2'] == 10.0)

    # Distribution 1: circle param = 2.0, ellipse param = 20.0
    assert all(dist_1_rows['circle__param1'] == 2.0)
    assert all(dist_1_rows['ellipse__param2'] == 20.0)

    # Distribution 2: circle param = 3.0, ellipse param = 30.0
    assert all(dist_2_rows['circle__param1'] == 3.0)
    assert all(dist_2_rows['ellipse__param2'] == 30.0)
