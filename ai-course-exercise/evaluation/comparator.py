"""
Model comparison utilities.
"""

import numpy as np


def compare_predictions(y_pred_custom: np.ndarray, y_pred_sklearn: np.ndarray,
                       y_true: np.ndarray, class_names: list) -> dict:
    """Compare predictions from both implementations."""
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)

    matches = np.sum(y_pred_custom == y_pred_sklearn)
    total = len(y_pred_custom)
    match_percentage = (matches / total) * 100

    print(f"\nPredictions match: {matches}/{total} ({match_percentage:.2f}%)")

    mismatches = np.where(y_pred_custom != y_pred_sklearn)[0]

    if len(mismatches) > 0:
        print(f"\nMismatched predictions: {len(mismatches)}")
        print("\nDetails of mismatches:")
        for idx in mismatches:
            print(f"  Sample {idx}: Custom={class_names[y_pred_custom[idx]]}, "
                  f"Sklearn={class_names[y_pred_sklearn[idx]]}, "
                  f"True={class_names[y_true[idx]]}")
    else:
        print("\nAll predictions match perfectly!")

    custom_correct = np.sum(y_pred_custom == y_true)
    sklearn_correct = np.sum(y_pred_sklearn == y_true)

    print(f"\nCorrect predictions:")
    print(f"  Custom:  {custom_correct}/{total} ({custom_correct/total*100:.2f}%)")
    print(f"  Sklearn: {sklearn_correct}/{total} ({sklearn_correct/total*100:.2f}%)")
    print("="*60)

    return {
        'total_predictions': total,
        'matching_predictions': matches,
        'match_percentage': match_percentage,
        'mismatches': mismatches,
        'custom_correct': custom_correct,
        'sklearn_correct': sklearn_correct
    }


def compare_parameters(model_custom, model_sklearn,
                      feature_names: list, class_names: list) -> dict:
    """Compare learned parameters between implementations."""
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)

    print("\nPrior Probabilities:")
    print(f"{'Class':<20} {'Custom':<15} {'Sklearn':<15} {'Difference':<15}")
    print("-" * 65)

    prior_differences = []
    for idx, class_name in enumerate(class_names):
        custom_prior = model_custom.priors_[idx]
        sklearn_prior = model_sklearn.priors_[idx]
        diff = abs(custom_prior - sklearn_prior)
        prior_differences.append(diff)
        print(f"{class_name:<20} {custom_prior:<15.6f} {sklearn_prior:<15.6f} {diff:<15.6e}")

    print("\n\nFeature Means:")
    mean_differences = []

    for class_idx, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  {'Feature':<20} {'Custom':<12} {'Sklearn':<12} {'Difference':<12}")
        print("  " + "-" * 56)

        for feat_idx, feat_name in enumerate(feature_names):
            custom_mean = model_custom.means_[class_idx, feat_idx]
            sklearn_mean = model_sklearn.means_[class_idx, feat_idx]
            diff = abs(custom_mean - sklearn_mean)
            mean_differences.append(diff)
            print(f"  {feat_name:<20} {custom_mean:<12.4f} {sklearn_mean:<12.4f} {diff:<12.4e}")

    print("\n\nSummary of Differences:")
    print(f"  Prior differences    - Mean: {np.mean(prior_differences):.6e}, "
          f"Max: {np.max(prior_differences):.6e}")
    print(f"  Mean differences     - Mean: {np.mean(mean_differences):.6e}, "
          f"Max: {np.max(mean_differences):.6e}")
    print("="*60)

    return {
        'prior_differences': prior_differences,
        'mean_differences': mean_differences
    }
