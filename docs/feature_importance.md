# Feature Importance Analysis

Lorepy provides statistical assessment of how strongly your x-feature is associated with the class distribution using the `feature_importance` function. This uses **permutation-based feature importance** with **log loss (cross-entropy)** as the scoring metric to test whether the relationship you see in your loreplot is statistically significant. Log loss evaluates the full predicted probability distribution rather than just hard class predictions, making it well-suited for lorepy's probability-based visualizations.

## How it Works

The function uses a robust resampling approach combined with sklearn's optimized permutation importance:

1. **Bootstrap/Jackknife Sampling**: Creates multiple subsamples of your data (default: 100 iterations)
2. **Permutation Importance**: For each subsample, uses sklearn's `permutation_importance` with proper cross-validation to avoid data leakage
3. **Feature Shuffling**: Randomly permutes the x-feature values while keeping confounders intact
4. **Performance Assessment**: Measures log loss increase using statistically sound train/test splits
5. **Statistical Summary**: Aggregates results across all iterations to provide confidence intervals and significance testing

This approach works with **any sklearn classifier** (LogisticRegression, SVM, RandomForest, etc.) and properly handles confounders by keeping them constant during shuffling. The implementation uses sklearn's battle-tested permutation importance algorithm for reliable, unbiased results.

```python
from lorepy import feature_importance

# Basic usage
stats = feature_importance(data=iris_df, x="sepal width (cm)", y="species", iterations=100)
print(stats['interpretation'])
# Output: "Feature importance: 0.2019 ± 0.0433. Positive in 100.0% of iterations (p=0.0000)"
```

## Understanding the Output

The function returns a dictionary with the following key statistics:

- **`mean_importance`**: Average log loss increase when x-feature is shuffled (higher = more important)
- **`std_importance`**: Standard deviation of importance across iterations
- **`importance_95ci_low/high`**: 95% confidence interval for the importance estimate
- **`mean_validation_log_loss`**: Mean log loss on the validation data across iterations (lower = better)
- **`std_validation_log_loss`**: Standard deviation of the validation log loss
- **`mean_permuted_log_loss`**: Mean log loss on the permuted data across iterations (lower = better)
- **`std_permuted_log_loss`**: Standard deviation of the permuted log loss
- **`proportion_positive`**: Fraction of iterations where importance > 0 (feature helps prediction)
- **`proportion_negative`**: Fraction of iterations where importance < 0 (feature hurts prediction)
- **`p_value`**: Empirical p-value for statistical significance (< 0.05 typically considered significant)
- **`interpretation`**: Human-readable summary of the results

## Advanced Usage

```python
from sklearn.svm import SVC

# With confounders and custom classifier
stats = feature_importance(
    data=data,
    x="age",
    y="disease",
    confounders=[("bmi", 25), ("sex", "female")],  # Control for these variables
    clf=SVC(probability=True),                      # Use SVM instead of logistic regression
    mode="jackknife",                              # Use jackknife instead of bootstrap
    iterations=200                                 # More iterations for precision
)

print(f"P-value: {stats['p_value']:.4f}")
print(f"95% CI: [{stats['importance_95ci_low']:.3f}, {stats['importance_95ci_high']:.3f}]")
```

## Interpretation Guidelines

- **Strong Association**: `p_value < 0.01`, `proportion_positive > 95%`
- **Moderate Association**: `p_value < 0.05`, `proportion_positive > 80%`
- **Weak/No Association**: `p_value > 0.05`, confidence interval includes zero
- **Negative Association**: `proportion_negative > proportion_positive` (unusual but possible)
