"""
Report generation utilities.
"""


def compare_metrics(metrics_custom: dict, metrics_sklearn: dict, class_names: list) -> None:
    """Compare evaluation metrics between implementations."""
    print("\n" + "="*60)
    print("METRICS COMPARISON")
    print("="*60)

    print("\nOverall Accuracy:")
    print(f"  Custom:  {metrics_custom['overall_accuracy']:.4f} "
          f"({metrics_custom['overall_accuracy']*100:.2f}%)")
    print(f"  Sklearn: {metrics_sklearn['overall_accuracy']:.4f} "
          f"({metrics_sklearn['overall_accuracy']*100:.2f}%)")
    print(f"  Difference: {abs(metrics_custom['overall_accuracy'] - metrics_sklearn['overall_accuracy']):.6f}")

    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Metric':<12} {'Custom':<12} {'Sklearn':<12} {'Difference':<12}")
    print("-" * 68)

    for idx, class_name in enumerate(class_names):
        custom_acc = metrics_custom['per_class_accuracy'][idx]
        sklearn_acc = metrics_sklearn['per_class_accuracy'][idx]
        print(f"{class_name:<20} {'Accuracy':<12} {custom_acc:<12.4f} {sklearn_acc:<12.4f} "
              f"{abs(custom_acc - sklearn_acc):<12.6f}")

        custom_prec = metrics_custom['precision'][idx]
        sklearn_prec = metrics_sklearn['precision'][idx]
        print(f"{'':<20} {'Precision':<12} {custom_prec:<12.4f} {sklearn_prec:<12.4f} "
              f"{abs(custom_prec - sklearn_prec):<12.6f}")

        custom_rec = metrics_custom['recall'][idx]
        sklearn_rec = metrics_sklearn['recall'][idx]
        print(f"{'':<20} {'Recall':<12} {custom_rec:<12.4f} {sklearn_rec:<12.4f} "
              f"{abs(custom_rec - sklearn_rec):<12.6f}")

        custom_f1 = metrics_custom['f1_score'][idx]
        sklearn_f1 = metrics_sklearn['f1_score'][idx]
        print(f"{'':<20} {'F1-Score':<12} {custom_f1:<12.4f} {sklearn_f1:<12.4f} "
              f"{abs(custom_f1 - sklearn_f1):<12.6f}")
        print()

    print("="*60)


def generate_comparison_summary(custom_results: dict, sklearn_results: dict,
                               class_names: list) -> str:
    """Generate a text summary of the comparison."""
    summary = []
    summary.append("\n" + "="*70)
    summary.append("COMPARISON SUMMARY")
    summary.append("="*70)

    summary.append("\n1. PREDICTION AGREEMENT")
    summary.append("-" * 70)
    custom_acc = custom_results['metrics']['overall_accuracy']
    sklearn_acc = sklearn_results['metrics']['overall_accuracy']

    summary.append(f"Overall Accuracy:")
    summary.append(f"  Custom Implementation:  {custom_acc:.4f} ({custom_acc*100:.2f}%)")
    summary.append(f"  Sklearn Implementation: {sklearn_acc:.4f} ({sklearn_acc*100:.2f}%)")
    summary.append(f"  Difference: {abs(custom_acc - sklearn_acc):.6f}")

    summary.append("\n2. KEY FINDINGS")
    summary.append("-" * 70)

    if abs(custom_acc - sklearn_acc) < 0.001:
        summary.append("[OK] Both implementations achieve nearly identical accuracy")
    else:
        summary.append(f"! Accuracy difference of {abs(custom_acc - sklearn_acc)*100:.2f}% detected")

    summary.append("\n3. IMPLEMENTATION CHARACTERISTICS")
    summary.append("-" * 70)
    summary.append("Custom NumPy Implementation:")
    summary.append("  + Full control over calculations")
    summary.append("  + Educational value - understand the math")
    summary.append("  + No external dependencies (except NumPy)")
    summary.append("  - May have numerical precision differences")
    summary.append("  - Requires more code")

    summary.append("\nScikit-learn Implementation:")
    summary.append("  + Highly optimized and tested")
    summary.append("  + Includes variance smoothing by default")
    summary.append("  + Easy to use, minimal code")
    summary.append("  + Well-documented and maintained")
    summary.append("  - Less educational (black box)")

    summary.append("\n" + "="*70)

    return "\n".join(summary)
