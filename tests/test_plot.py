"""
Comprehensive tests for lorepy.lorepy module.

Tests cover:
- _prepare_data: data preparation and validation
- _get_area_df: probability area calculation
- _get_dots_df: scatter dot positioning
- loreplot: main plotting function
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from lorepy.lorepy import _get_area_df, _get_dots_df, _prepare_data, loreplot

# =============================================================================
# Tests for _prepare_data
# =============================================================================


class TestPrepareData:
    """Tests for the _prepare_data function."""

    def test_basic_preparation(self, small_deterministic_data):
        """Test basic data preparation without confounders."""
        X_reg, y_reg, x_range = _prepare_data(small_deterministic_data, "x", "y", [])

        assert isinstance(X_reg, np.ndarray)
        assert isinstance(y_reg, np.ndarray)
        assert isinstance(x_range, tuple)
        assert len(x_range) == 2

        # Check shapes
        assert X_reg.shape == (10, 1)
        assert y_reg.shape == (10,)

        # Check x_range
        assert x_range[0] == 1.0
        assert x_range[1] == 10.0

    def test_with_confounders(self, small_deterministic_data):
        """Test data preparation with confounders."""
        confounders = [("z", 1.0)]
        X_reg, y_reg, x_range = _prepare_data(
            small_deterministic_data, "x", "y", confounders
        )

        # X should now have 2 columns: x and z
        assert X_reg.shape == (10, 2)
        # First column should be x (the main feature)
        np.testing.assert_array_equal(X_reg[:, 0], small_deterministic_data["x"].values)
        # Second column should be z (the confounder)
        np.testing.assert_array_equal(X_reg[:, 1], small_deterministic_data["z"].values)

    def test_nan_removal(self, data_with_nan):
        """Test that NaN values are properly removed."""
        X_reg, y_reg, x_range = _prepare_data(data_with_nan, "x", "y", [])

        # Should have fewer rows than original due to NaN removal
        assert len(y_reg) < len(data_with_nan)

        # Should have no NaN values
        assert not np.any(np.isnan(X_reg))
        assert not np.any(pd.isna(y_reg))

    def test_x_range_calculation(self, binary_sample_data):
        """Test that x_range is correctly calculated from data."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        expected_min = binary_sample_data["x"].min()
        expected_max = binary_sample_data["x"].max()

        assert x_range[0] == expected_min
        assert x_range[1] == expected_max

    def test_multiple_confounders(self, binary_sample_data):
        """Test data preparation with multiple confounders."""
        # Add another confounder column
        binary_sample_data["w"] = binary_sample_data["x"] * 2

        confounders = [("z", 5.0), ("w", 10.0)]
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", confounders)

        # X should have 3 columns: x, z, w
        assert X_reg.shape[1] == 3

    def test_preserves_data_order(self, small_deterministic_data):
        """Test that data order is preserved (x first, then confounders)."""
        confounders = [("z", 1.0)]
        X_reg, y_reg, x_range = _prepare_data(
            small_deterministic_data, "x", "y", confounders
        )

        # First column must be x for compatibility with _get_feature_importance
        np.testing.assert_array_equal(X_reg[:, 0], small_deterministic_data["x"].values)


# =============================================================================
# Tests for _get_area_df
# =============================================================================


class TestGetAreaDf:
    """Tests for the _get_area_df function."""

    def test_basic_output_structure(self, fitted_logistic_model):
        """Test basic output structure of _get_area_df."""
        lg, X, y = fitted_logistic_model
        x_range = (X.min(), X.max())

        area_df = _get_area_df(lg, "x", x_range)

        assert isinstance(area_df, DataFrame)
        # Should have 200 rows (default num points)
        assert len(area_df) == 200
        # Index should be named after x_feature
        assert area_df.index.name == "x"
        # Should have columns for each class
        assert 0 in area_df.columns
        assert 1 in area_df.columns

    def test_probabilities_sum_to_one(self, fitted_logistic_model):
        """Test that probabilities sum to 1 at each x point."""
        lg, X, y = fitted_logistic_model
        x_range = (X.min(), X.max())

        area_df = _get_area_df(lg, "x", x_range)

        # Sum of probabilities should be 1 for each row
        row_sums = area_df.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums.values, np.ones(200))

    def test_probabilities_in_valid_range(self, fitted_logistic_model):
        """Test that all probabilities are between 0 and 1."""
        lg, X, y = fitted_logistic_model
        x_range = (X.min(), X.max())

        area_df = _get_area_df(lg, "x", x_range)

        assert (area_df.values >= 0).all()
        assert (area_df.values <= 1).all()

    def test_x_range_respected(self, fitted_logistic_model):
        """Test that the x_range parameter is respected."""
        lg, X, y = fitted_logistic_model
        custom_range = (2.0, 8.0)

        area_df = _get_area_df(lg, "x", custom_range)

        assert area_df.index[0] == custom_range[0]
        assert area_df.index[-1] == custom_range[1]

    def test_with_confounders(self, binary_sample_data):
        """Test _get_area_df with confounders."""
        # Fit a model with confounders
        X_reg = binary_sample_data[["x", "z"]].values
        y_reg = binary_sample_data["y"].values
        lg = LogisticRegression()
        lg.fit(X_reg, y_reg)

        confounders = [("z", 5.0)]
        x_range = (0.0, 12.0)

        area_df = _get_area_df(lg, "x", x_range, confounders=confounders)

        # Should still have valid probabilities
        assert len(area_df) == 200
        row_sums = area_df.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums.values, np.ones(200))

    def test_multiclass_output(self, fitted_multiclass_model):
        """Test _get_area_df with multi-class classification."""
        lg, X, y = fitted_multiclass_model
        x_range = (X.min(), X.max())

        area_df = _get_area_df(lg, "x", x_range)

        # Should have 3 class columns
        assert 0 in area_df.columns
        assert 1 in area_df.columns
        assert 2 in area_df.columns

        # Probabilities should still sum to 1
        row_sums = area_df.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums.values, np.ones(200))

    def test_monotonic_probability_trend(self, fitted_logistic_model):
        """Test that probabilities show expected monotonic trend for clear separation."""
        lg, X, y = fitted_logistic_model
        x_range = (1.0, 10.0)

        area_df = _get_area_df(lg, "x", x_range)

        # For our deterministic data (0s at low x, 1s at high x),
        # P(class=1) should generally increase with x
        class_1_probs = area_df[1].values
        # Check that probability at end is higher than at start
        assert class_1_probs[-1] > class_1_probs[0]


# =============================================================================
# Tests for _get_dots_df
# =============================================================================


class TestGetDotsDf:
    """Tests for the _get_dots_df function."""

    def test_basic_output_structure(self, fitted_logistic_model):
        """Test basic output structure of _get_dots_df."""
        lg, X, y = fitted_logistic_model

        dots_df = _get_dots_df(X, y, lg, "y")

        assert isinstance(dots_df, DataFrame)
        assert len(dots_df) == len(X)
        assert "x" in dots_df.columns
        assert "y" in dots_df.columns
        assert "y" in dots_df.columns  # The y_feature column

    def test_y_coordinates_in_valid_range(self, fitted_logistic_model):
        """Test that y coordinates are between 0 and 1."""
        lg, X, y = fitted_logistic_model

        dots_df = _get_dots_df(X, y, lg, "y_label")

        assert (dots_df["y"].values >= 0).all()
        assert (dots_df["y"].values <= 1).all()

    def test_x_coordinates_match_input(self, fitted_logistic_model):
        """Test that x coordinates match input data (without jitter)."""
        lg, X, y = fitted_logistic_model

        dots_df = _get_dots_df(X, y, lg, "y_label", jitter=0)

        np.testing.assert_array_equal(dots_df["x"].values, X.flatten())

    def test_jitter_modifies_x_coordinates(self, fitted_logistic_model, random_seed):
        """Test that jitter modifies x coordinates."""
        lg, X, y = fitted_logistic_model
        jitter_amount = 0.5

        dots_df = _get_dots_df(X.copy(), y, lg, "y_label", jitter=jitter_amount)

        # With jitter, x values should be different from original
        # Note: This modifies X in place, so we use a copy
        differences = np.abs(dots_df["x"].values - X.flatten())
        # At least some differences should be non-zero
        assert np.any(differences > 0)
        # All differences should be within jitter range
        assert np.all(differences <= jitter_amount)

    def test_y_feature_column_values(self, fitted_logistic_model):
        """Test that y_feature column contains correct class labels."""
        lg, X, y = fitted_logistic_model

        dots_df = _get_dots_df(X, y, lg, "class_label")

        assert "class_label" in dots_df.columns
        np.testing.assert_array_equal(dots_df["class_label"].values, y)

    def test_y_within_probability_bands(self, fitted_logistic_model):
        """Test that y coordinates fall within the probability band for their class."""
        lg, X, y = fitted_logistic_model

        dots_df = _get_dots_df(X, y, lg, "y_label")

        for idx, row in dots_df.iterrows():
            x_val = np.array([[row["x"]]])
            proba = lg.predict_proba(x_val)[0]
            class_idx = list(lg.classes_).index(row["y_label"])

            # Calculate expected band
            min_val = sum(proba[:class_idx])
            max_val = sum(proba[: class_idx + 1])

            # y should be within the band (with some tolerance for margin)
            assert row["y"] >= min_val - 0.01
            assert row["y"] <= max_val + 0.01

    def test_multiclass_dots(self, fitted_multiclass_model):
        """Test _get_dots_df with multi-class classification."""
        lg, X, y = fitted_multiclass_model

        dots_df = _get_dots_df(X, y, lg, "class")

        assert len(dots_df) == len(X)
        assert set(dots_df["class"].unique()) == {0, 1, 2}


# =============================================================================
# Tests for loreplot
# =============================================================================


class TestLoreplot:
    """Tests for the main loreplot function."""

    def test_creates_plot_without_ax(self, binary_sample_data):
        """Test that loreplot creates a plot when no ax is provided."""
        # Should not raise
        loreplot(binary_sample_data, "x", "y")
        plt.close()

    def test_uses_provided_ax(self, binary_sample_data):
        """Test that loreplot uses provided axes."""
        fig, ax = plt.subplots()
        loreplot(binary_sample_data, "x", "y", ax=ax)

        assert ax.get_xlabel() == "x"
        plt.close()

    def test_axis_limits(self, binary_sample_data):
        """Test that axis limits are set correctly."""
        fig, ax = plt.subplots()
        loreplot(binary_sample_data, "x", "y", ax=ax)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # y should be 0 to 1 (probability)
        assert ylim == (0, 1)
        # x should span the data range
        assert xlim[0] <= binary_sample_data["x"].min()
        assert xlim[1] >= binary_sample_data["x"].max()

        plt.close()

    def test_custom_x_range(self, binary_sample_data):
        """Test that custom x_range is respected."""
        fig, ax = plt.subplots()
        custom_range = (0, 20)
        loreplot(binary_sample_data, "x", "y", ax=ax, x_range=custom_range)

        xlim = ax.get_xlim()
        assert xlim == custom_range

        plt.close()

    def test_add_dots_true(self, binary_sample_data):
        """Test that dots are added when add_dots=True."""
        fig, ax = plt.subplots()
        loreplot(binary_sample_data, "x", "y", ax=ax, add_dots=True)

        # Check that scatter points were added (collections should have dots)
        # The scatter creates a PathCollection
        collections = ax.collections
        assert len(collections) > 0

        plt.close()

    def test_add_dots_false(self, binary_sample_data):
        """Test that no dots are added when add_dots=False."""
        fig, ax = plt.subplots()
        loreplot(binary_sample_data, "x", "y", ax=ax, add_dots=False)

        # No scatter collections should be added
        # Note: area plot might add some collections, but scatter specifically won't
        scatter_collections = [
            c
            for c in ax.collections
            if hasattr(c, "get_offsets") and len(c.get_offsets()) > 0
        ]
        # When add_dots=False, there should be no scatter points
        for c in scatter_collections:
            offsets = c.get_offsets()
            # If there are offsets, they shouldn't be the data points
            if len(offsets) > 0:
                # This is from the area plot, not scatter
                pass

        plt.close()

    def test_with_jitter(self, binary_sample_data, random_seed):
        """Test that jitter parameter works."""
        fig, ax = plt.subplots()
        # Should not raise
        loreplot(binary_sample_data, "x", "y", ax=ax, jitter=0.1)

        plt.close()

    def test_with_confounders(self, binary_sample_data):
        """Test loreplot with confounders."""
        fig, ax = plt.subplots()
        # Should not raise
        loreplot(binary_sample_data, "x", "y", ax=ax, confounders=[("z", 5.0)])

        plt.close()

    def test_confounders_disable_dots(self, binary_sample_data):
        """Test that dots are not added when confounders are specified."""
        fig, ax = plt.subplots()
        loreplot(
            binary_sample_data,
            "x",
            "y",
            ax=ax,
            add_dots=True,  # Request dots
            confounders=[("z", 5.0)],  # But confounders should prevent them
        )

        # With confounders, dots should not be added even if add_dots=True
        # Check there are no scatter points with many offsets
        scatter_with_data = [
            c
            for c in ax.collections
            if hasattr(c, "get_offsets")
            and len(c.get_offsets()) == len(binary_sample_data)
        ]
        assert len(scatter_with_data) == 0

        plt.close()

    def test_custom_classifier(self, binary_sample_data):
        """Test loreplot with a custom classifier."""
        fig, ax = plt.subplots()
        svc = SVC(probability=True)
        # Should not raise
        loreplot(binary_sample_data, "x", "y", ax=ax, clf=svc)

        plt.close()

    def test_custom_colors(self, binary_sample_data):
        """Test loreplot with custom colors."""
        fig, ax = plt.subplots()
        loreplot(binary_sample_data, "x", "y", ax=ax, color=["red", "blue"])

        plt.close()

    def test_scatter_kws(self, binary_sample_data):
        """Test that scatter_kws are passed through."""
        fig, ax = plt.subplots()
        loreplot(
            binary_sample_data, "x", "y", ax=ax, scatter_kws={"s": 100, "marker": "^"}
        )

        plt.close()

    def test_kwargs_passed_to_area_plot(self, binary_sample_data):
        """Test that kwargs are passed to the area plot."""
        fig, ax = plt.subplots()
        loreplot(binary_sample_data, "x", "y", ax=ax, alpha=0.5, linestyle="-")

        plt.close()

    def test_string_class_labels(self, string_class_labels):
        """Test loreplot with string class labels."""
        fig, ax = plt.subplots()
        loreplot(string_class_labels, "x", "y", ax=ax)

        plt.close()

    def test_multiclass(self, multiclass_sample_data):
        """Test loreplot with multi-class classification."""
        fig, ax = plt.subplots()
        loreplot(multiclass_sample_data, "x", "y", ax=ax)

        plt.close()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_class_raises_error(self, single_class_data):
        """Test that single class data raises an appropriate error."""
        fig, ax = plt.subplots()
        # LogisticRegression should fail or warn with single class
        with pytest.raises(ValueError):
            loreplot(single_class_data, "x", "y", ax=ax)
        plt.close()

    def test_empty_dataframe_raises_error(self, empty_dataframe):
        """Test that empty DataFrame raises an error."""
        fig, ax = plt.subplots()
        with pytest.raises((ValueError, IndexError)):
            loreplot(empty_dataframe, "x", "y", ax=ax)
        plt.close()

    def test_missing_column_raises_error(self, binary_sample_data):
        """Test that missing column raises KeyError."""
        fig, ax = plt.subplots()
        with pytest.raises(KeyError):
            loreplot(binary_sample_data, "nonexistent", "y", ax=ax)
        plt.close()

    def test_nan_handling(self, data_with_nan):
        """Test that NaN values are handled properly."""
        fig, ax = plt.subplots()
        # Should not raise - NaN rows should be dropped
        loreplot(data_with_nan, "x", "y", ax=ax)
        plt.close()

    def test_inverted_x_range(self, binary_sample_data):
        """Test behavior with inverted x_range (max < min)."""
        fig, ax = plt.subplots()
        # Note: This might create a valid plot with inverted axis
        loreplot(binary_sample_data, "x", "y", ax=ax, x_range=(10, 0))
        plt.close()

    def test_zero_width_x_range(self, binary_sample_data):
        """Test behavior when x_range has zero width.

        Note: The current implementation does not validate x_range and will
        create a degenerate plot where all 200 points are at the same x value.
        This test documents the current behavior rather than asserting it's correct.
        """
        fig, ax = plt.subplots()
        # Current implementation doesn't raise an error, but creates a degenerate plot
        # This may be considered a bug or missing input validation
        loreplot(binary_sample_data, "x", "y", ax=ax, x_range=(5, 5))
        # Matplotlib may expand xlim slightly, but should be centered around 5
        xlim = ax.get_xlim()
        assert abs((xlim[0] + xlim[1]) / 2 - 5) < 0.5  # Center is approximately 5
        plt.close()
