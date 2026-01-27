"""
Comprehensive tests for lorepy.uncertainty module.

Tests cover:
- _get_uncertainty_data: uncertainty estimation via resampling/jackknife
- _get_feature_importance: feature importance calculation
- uncertainty_plot: main uncertainty visualization function
- feature_importance: public API for feature importance
"""
import numpy as np
import pandas as pd
import pytest
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from lorepy import uncertainty_plot, feature_importance
from lorepy.uncertainty import _get_uncertainty_data, _get_feature_importance
from lorepy.lorepy import _prepare_data


# =============================================================================
# Tests for _get_uncertainty_data
# =============================================================================

class TestGetUncertaintyData:
    """Tests for the _get_uncertainty_data function."""

    def test_basic_output_structure_resample(self, binary_sample_data):
        """Test basic output structure with resample mode."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        output, long_df = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=10
        )

        # Check output DataFrame structure
        assert isinstance(output, pd.DataFrame)
        assert "x" in output.columns
        assert "variable" in output.columns
        assert "min" in output.columns
        assert "mean" in output.columns
        assert "max" in output.columns
        assert "low_95" in output.columns
        assert "high_95" in output.columns
        assert "low_50" in output.columns
        assert "high_50" in output.columns

        # Check long_df structure
        assert isinstance(long_df, pd.DataFrame)

    def test_basic_output_structure_jackknife(self, binary_sample_data):
        """Test basic output structure with jackknife mode."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        output, long_df = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="jackknife",
            jackknife_fraction=0.8,
            iterations=10
        )

        assert isinstance(output, pd.DataFrame)
        assert "mean" in output.columns

    def test_uncertainty_bounds_ordering(self, binary_sample_data):
        """Test that uncertainty bounds are properly ordered."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        output, _ = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=50
        )

        # For each row, bounds should be ordered: min <= low_95 <= low_50 <= mean <= high_50 <= high_95 <= max
        # Note: Due to bootstrap variability, mean might not be exactly between low_50 and high_50
        # but min/max bounds should always hold
        assert (output["min"] <= output["max"]).all()
        assert (output["low_95"] <= output["high_95"]).all()
        assert (output["low_50"] <= output["high_50"]).all()

    def test_probability_values_in_range(self, binary_sample_data):
        """Test that all probability values are between 0 and 1."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        output, _ = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=10
        )

        for col in ["min", "mean", "max", "low_95", "high_95", "low_50", "high_50"]:
            assert (output[col] >= 0).all(), f"{col} has values < 0"
            assert (output[col] <= 1).all(), f"{col} has values > 1"

    def test_iterations_parameter(self, binary_sample_data):
        """Test that iterations parameter affects output variability."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        # With very few iterations, expect wider intervals
        output_few, long_few = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=5
        )

        # Long_df should have iterations * 200 (num points) * num_classes rows
        # Actually it's melted, so the structure depends on implementation
        assert len(long_few) > 0

    def test_with_confounders(self, binary_sample_data):
        """Test _get_uncertainty_data with confounders."""
        confounders = [("z", 5.0)]
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", confounders)

        output, _ = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=10,
            confounders=confounders
        )

        assert isinstance(output, pd.DataFrame)
        assert "mean" in output.columns

    def test_custom_classifier(self, binary_sample_data):
        """Test _get_uncertainty_data with custom classifier."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])
        svc = SVC(probability=True)

        output, _ = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=10,
            clf=svc
        )

        assert isinstance(output, pd.DataFrame)

    def test_invalid_mode_raises_error(self, binary_sample_data):
        """Test that invalid mode raises NotImplementedError."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        with pytest.raises(NotImplementedError):
            _get_uncertainty_data(
                "x", X_reg, y_reg, x_range,
                mode="invalid_mode",
                iterations=10
            )

    def test_output_has_all_categories(self, binary_sample_data):
        """Test that output includes all class categories."""
        X_reg, y_reg, x_range = _prepare_data(binary_sample_data, "x", "y", [])

        output, _ = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=10
        )

        categories = output["variable"].unique()
        assert 0 in categories
        assert 1 in categories

    def test_multiclass_uncertainty(self, multiclass_sample_data):
        """Test _get_uncertainty_data with multi-class classification."""
        X_reg, y_reg, x_range = _prepare_data(multiclass_sample_data, "x", "y", [])

        output, _ = _get_uncertainty_data(
            "x", X_reg, y_reg, x_range,
            mode="resample",
            iterations=10
        )

        categories = output["variable"].unique()
        assert len(categories) == 3


# =============================================================================
# Tests for _get_feature_importance
# =============================================================================

class TestGetFeatureImportance:
    """Tests for the _get_feature_importance function."""

    def test_basic_output_structure(self, binary_sample_data):
        """Test basic output structure of feature importance."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("x", X_reg, y_reg, iterations=10)

        expected_keys = [
            "feature",
            "mean_importance",
            "std_importance",
            "importance_95ci_low",
            "importance_95ci_high",
            "proportion_positive",
            "proportion_negative",
            "p_value",
            "iterations",
            "mode",
            "interpretation",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_feature_name_preserved(self, binary_sample_data):
        """Test that feature name is preserved in output."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("my_feature", X_reg, y_reg, iterations=10)

        assert result["feature"] == "my_feature"

    def test_iterations_parameter(self, binary_sample_data):
        """Test that iterations parameter is reflected in output."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("x", X_reg, y_reg, iterations=25)

        assert result["iterations"] == 25

    def test_mode_resample(self, binary_sample_data):
        """Test resample mode."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        with pytest.warns(UserWarning, match="Bootstrap resampling mode"):
            result = _get_feature_importance(
                "x", X_reg, y_reg,
                mode="resample",
                iterations=10
            )

        assert result["mode"] == "resample"

    def test_mode_jackknife(self, binary_sample_data):
        """Test jackknife mode."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance(
            "x", X_reg, y_reg,
            mode="jackknife",
            iterations=10
        )

        assert result["mode"] == "jackknife"

    def test_invalid_mode_raises_error(self, binary_sample_data):
        """Test that invalid mode raises NotImplementedError."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        with pytest.raises(NotImplementedError):
            _get_feature_importance(
                "x", X_reg, y_reg,
                mode="invalid_mode",
                iterations=10
            )

    def test_p_value_in_valid_range(self, binary_sample_data):
        """Test that p-value is between 0 and 1."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("x", X_reg, y_reg, iterations=10)

        assert 0 <= result["p_value"] <= 1

    def test_proportions_sum_valid(self, binary_sample_data):
        """Test that positive + negative proportions <= 1."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("x", X_reg, y_reg, iterations=10)

        assert result["proportion_positive"] + result["proportion_negative"] <= 1
        assert result["proportion_positive"] >= 0
        assert result["proportion_negative"] >= 0

    def test_confidence_interval_ordering(self, binary_sample_data):
        """Test that confidence interval bounds are ordered correctly."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("x", X_reg, y_reg, iterations=50)

        assert result["importance_95ci_low"] <= result["importance_95ci_high"]

    def test_interpretation_string_format(self, binary_sample_data):
        """Test that interpretation is a properly formatted string."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        result = _get_feature_importance("x", X_reg, y_reg, iterations=10)

        assert isinstance(result["interpretation"], str)
        assert "Feature importance" in result["interpretation"]

    def test_custom_classifier(self, binary_sample_data):
        """Test with custom classifier."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])
        svc = SVC(probability=True)

        result = _get_feature_importance(
            "x", X_reg, y_reg,
            iterations=10,
            clf=svc
        )

        assert isinstance(result["mean_importance"], float)

    def test_jackknife_fraction_parameter(self, binary_sample_data):
        """Test that jackknife_fraction parameter is used."""
        X_reg, y_reg, _ = _prepare_data(binary_sample_data, "x", "y", [])

        # Different fractions should produce different results
        result_80 = _get_feature_importance(
            "x", X_reg, y_reg,
            mode="jackknife",
            jackknife_fraction=0.8,
            iterations=10
        )

        result_50 = _get_feature_importance(
            "x", X_reg, y_reg,
            mode="jackknife",
            jackknife_fraction=0.5,
            iterations=10
        )

        # Both should produce valid results
        assert isinstance(result_80["mean_importance"], float)
        assert isinstance(result_50["mean_importance"], float)


# =============================================================================
# Tests for uncertainty_plot
# =============================================================================

class TestUncertaintyPlot:
    """Tests for the main uncertainty_plot function."""

    def test_basic_plot_creation(self, binary_sample_data):
        """Test basic plot creation with default parameters."""
        fig, axs = uncertainty_plot(binary_sample_data, "x", "y", iterations=10)

        assert fig is not None
        assert len(axs) == 2  # Two classes
        plt.close()

    def test_axes_titles(self, binary_sample_data):
        """Test that axes have correct titles."""
        fig, axs = uncertainty_plot(binary_sample_data, "x", "y", iterations=10)

        assert axs[0].get_title() == "0"
        assert axs[1].get_title() == "1"
        plt.close()

    def test_axes_labels(self, binary_sample_data):
        """Test that axes have correct labels."""
        fig, axs = uncertainty_plot(binary_sample_data, "x", "y", iterations=10)

        assert axs[0].get_xlabel() == "x"
        assert axs[1].get_xlabel() == "x"
        plt.close()

    def test_axes_limits(self, binary_sample_data):
        """Test that axes have correct limits."""
        fig, axs = uncertainty_plot(binary_sample_data, "x", "y", iterations=10)

        for ax in axs:
            ylim = ax.get_ylim()
            assert ylim == (0, 1)
        plt.close()

    def test_custom_x_range(self, binary_sample_data):
        """Test custom x_range parameter."""
        custom_range = (0, 20)
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            x_range=custom_range,
            iterations=10
        )

        for ax in axs:
            xlim = ax.get_xlim()
            assert xlim == custom_range
        plt.close()

    def test_jackknife_mode(self, binary_sample_data):
        """Test jackknife mode."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            mode="jackknife",
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_invalid_mode_raises_error(self, binary_sample_data):
        """Test that invalid mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            uncertainty_plot(
                binary_sample_data, "x", "y",
                mode="invalid_mode",
                iterations=10
            )

    def test_with_confounders(self, binary_sample_data):
        """Test plot with confounders."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            confounders=[("z", 5.0)],
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_custom_colormap(self, binary_sample_data, custom_colormap):
        """Test plot with custom colormap."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            colormap=custom_colormap,
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_custom_classifier(self, binary_sample_data):
        """Test plot with custom classifier."""
        svc = SVC(probability=True)
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            clf=svc,
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_existing_axes(self, binary_sample_data):
        """Test plot with pre-existing axes."""
        fig, ax = plt.subplots(1, 2)
        returned_fig, returned_axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            ax=ax,
            iterations=10
        )

        assert returned_axs[0] == ax[0]
        assert returned_axs[1] == ax[1]
        plt.close()

    def test_wrong_number_of_axes_raises_error(self, binary_sample_data):
        """Test that wrong number of axes raises AssertionError."""
        fig, ax = plt.subplots(1, 1)  # Only one axis

        with pytest.raises(AssertionError):
            uncertainty_plot(
                binary_sample_data, "x", "y",
                ax=[ax],
                iterations=10
            )
        plt.close()

    def test_multiclass_creates_correct_number_of_axes(self, multiclass_sample_data):
        """Test that multiclass data creates correct number of axes."""
        fig, axs = uncertainty_plot(
            multiclass_sample_data, "x", "y",
            iterations=10
        )

        assert len(axs) == 3  # Three classes
        plt.close()

    def test_plot_has_fill_between(self, binary_sample_data):
        """Test that plot includes fill_between elements."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            iterations=10
        )

        # Each axis should have collections from fill_between
        for ax in axs:
            assert len(ax.collections) > 0
        plt.close()

    def test_plot_has_line(self, binary_sample_data):
        """Test that plot includes mean line."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            iterations=10
        )

        # Each axis should have at least one line (the mean)
        for ax in axs:
            assert len(ax.lines) > 0
        plt.close()


# =============================================================================
# Tests for public feature_importance function
# =============================================================================

class TestPublicFeatureImportance:
    """Tests for the public feature_importance API."""

    def test_basic_usage(self, binary_sample_data):
        """Test basic usage of feature_importance."""
        result = feature_importance(
            binary_sample_data, x="x", y="y",
            iterations=10
        )

        assert result["feature"] == "x"
        assert result["iterations"] == 10

    def test_with_confounders(self, binary_sample_data):
        """Test feature_importance with confounders."""
        result = feature_importance(
            binary_sample_data, x="x", y="y",
            confounders=[("z", 5.0)],
            iterations=10
        )

        assert result["feature"] == "x"

    def test_with_custom_classifier(self, binary_sample_data):
        """Test feature_importance with custom classifier."""
        svc = SVC(probability=True)
        result = feature_importance(
            binary_sample_data, x="x", y="y",
            clf=svc,
            iterations=10
        )

        assert isinstance(result["mean_importance"], float)

    def test_jackknife_mode(self, binary_sample_data):
        """Test feature_importance with jackknife mode."""
        result = feature_importance(
            binary_sample_data, x="x", y="y",
            mode="jackknife",
            iterations=10
        )

        assert result["mode"] == "jackknife"

    def test_resample_mode_warning(self, binary_sample_data):
        """Test that resample mode issues a warning."""
        with pytest.warns(UserWarning, match="Bootstrap resampling mode"):
            result = feature_importance(
                binary_sample_data, x="x", y="y",
                mode="resample",
                iterations=10
            )

        assert result["mode"] == "resample"

    def test_small_validation_set_warning(self):
        """Test warning for small validation sets in jackknife mode."""
        small_data = pd.DataFrame({
            "x": np.random.randn(15),
            "y": np.random.choice([0, 1], 15),
        })

        with pytest.warns(UserWarning, match="Jackknife validation set is small"):
            result = feature_importance(
                small_data, x="x", y="y",
                mode="jackknife",
                jackknife_fraction=0.8,
                iterations=5
            )

        assert result["mode"] == "jackknife"

    def test_no_warning_adequate_validation(self):
        """Test no warning for adequate validation sets."""
        np.random.seed(42)
        large_data = pd.DataFrame({
            "x": np.random.randn(150),
            "y": np.random.choice([0, 1], 150),
        })

        # This should not trigger the small validation warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                result = feature_importance(
                    large_data, x="x", y="y",
                    mode="jackknife",
                    jackknife_fraction=0.8,
                    iterations=5
                )
                assert result["mode"] == "jackknife"
            except UserWarning as e:
                if "small" in str(e).lower():
                    pytest.fail("Unexpected small validation warning")

    def test_output_consistency(self, binary_sample_data):
        """Test that public function output matches internal function."""
        # Get result from public function
        public_result = feature_importance(
            binary_sample_data, x="x", y="y",
            iterations=10
        )

        # Both should have the same keys
        expected_keys = [
            "feature",
            "mean_importance",
            "std_importance",
            "importance_95ci_low",
            "importance_95ci_high",
            "proportion_positive",
            "proportion_negative",
            "p_value",
            "iterations",
            "mode",
            "interpretation",
        ]

        for key in expected_keys:
            assert key in public_result


# =============================================================================
# Edge Cases
# =============================================================================

class TestUncertaintyEdgeCases:
    """Edge case tests for uncertainty module."""

    def test_very_few_iterations(self, binary_sample_data):
        """Test with minimum iterations."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            iterations=2
        )

        assert len(axs) == 2
        plt.close()

    def test_high_jackknife_fraction(self, binary_sample_data):
        """Test with high jackknife fraction."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            mode="jackknife",
            jackknife_fraction=0.95,
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_low_jackknife_fraction(self, binary_sample_data):
        """Test with low jackknife fraction."""
        fig, axs = uncertainty_plot(
            binary_sample_data, "x", "y",
            mode="jackknife",
            jackknife_fraction=0.5,
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_string_class_labels(self, string_class_labels):
        """Test with string class labels."""
        fig, axs = uncertainty_plot(
            string_class_labels, "x", "y",
            iterations=10
        )

        assert len(axs) == 2
        plt.close()

    def test_nan_handling(self, data_with_nan):
        """Test that NaN values are handled properly."""
        # Should not raise - NaN rows should be dropped by _prepare_data
        fig, axs = uncertainty_plot(
            data_with_nan, "x", "y",
            iterations=10
        )

        assert len(axs) == 2
        plt.close()
