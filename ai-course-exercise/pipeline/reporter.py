"""
Report generation pipeline.
"""

import os
from datetime import datetime
from evaluation import generate_comparison_summary


def save_summary_report(results, class_names, pred_comparison):
    """Generate and save summary report."""
    custom_results = results['custom']
    sklearn_results = results['sklearn']

    summary = generate_comparison_summary(custom_results, sklearn_results, class_names)
    print(summary)

    summary_file = os.path.join("outputs", "comparison_summary.txt")
    os.makedirs("outputs", exist_ok=True)

    with open(summary_file, 'w') as f:
        f.write(f"Naive Bayes IRIS Classification - Comparison Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(summary)
        f.write("\n\n" + "="*70 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("="*70 + "\n\n")

        metrics_custom = custom_results['metrics']
        metrics_sklearn = sklearn_results['metrics']

        f.write("CUSTOM IMPLEMENTATION METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Overall Accuracy: {metrics_custom['overall_accuracy']:.4f}\n")
        for idx, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Accuracy:  {metrics_custom['per_class_accuracy'][idx]:.4f}\n")
            f.write(f"  Precision: {metrics_custom['precision'][idx]:.4f}\n")
            f.write(f"  Recall:    {metrics_custom['recall'][idx]:.4f}\n")
            f.write(f"  F1-Score:  {metrics_custom['f1_score'][idx]:.4f}\n")

        f.write("\n\nSKLEARN IMPLEMENTATION METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Overall Accuracy: {metrics_sklearn['overall_accuracy']:.4f}\n")
        for idx, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Accuracy:  {metrics_sklearn['per_class_accuracy'][idx]:.4f}\n")
            f.write(f"  Precision: {metrics_sklearn['precision'][idx]:.4f}\n")
            f.write(f"  Recall:    {metrics_sklearn['recall'][idx]:.4f}\n")
            f.write(f"  F1-Score:  {metrics_sklearn['f1_score'][idx]:.4f}\n")

        f.write("\n\nCONFUSION MATRICES:\n")
        f.write("-" * 70 + "\n")
        f.write("\nCustom Implementation:\n")
        f.write(str(metrics_custom['confusion_matrix']))
        f.write("\n\nScikit-learn Implementation:\n")
        f.write(str(metrics_sklearn['confusion_matrix']))

    print(f"\n[OK] Summary report saved to: {summary_file}")


def print_final_summary(results, pred_comparison, plots_dir):
    """Print final execution summary."""
    metrics_custom = results['custom']['metrics']
    metrics_sklearn = results['sklearn']['metrics']

    print("\nGenerated Files:")
    print(f"  * Plots: {plots_dir}/ directory")
    print(f"     - feature_distributions.png")
    print(f"     - feature_relationships.png")
    print(f"     - confusion_matrix_custom.png")
    print(f"     - confusion_matrix_sklearn.png")
    print(f"     - confusion_matrices_comparison.png")
    print(f"     - accuracy_comparison.png")
    print(f"     - learned_distributions.png")
    print(f"\n  * Reports: outputs/ directory")
    print(f"     - comparison_summary.txt")

    print("\nKey Results:")
    print(f"  Custom Implementation Accuracy:  {metrics_custom['overall_accuracy']*100:.2f}%")
    print(f"  Sklearn Implementation Accuracy: {metrics_sklearn['overall_accuracy']*100:.2f}%")
    print(f"  Prediction Agreement: {pred_comparison['match_percentage']:.2f}%")

    print(f"\n[OK] All tasks completed successfully!")
    print(f"Execution ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
