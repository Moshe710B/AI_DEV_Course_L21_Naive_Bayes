"""
Main execution script for Naive Bayes IRIS Classification.

This script orchestrates the complete pipeline for comparing
custom NumPy and scikit-learn implementations of Gaussian Naive Bayes.
"""

import os
import sys
import numpy as np
from datetime import datetime

from data import prepare_data
from evaluation import compare_predictions, compare_parameters, compare_metrics
from pipeline import (
    print_header, train_and_evaluate_models,
    create_exploratory_visualizations, create_comparison_visualizations,
    save_summary_report, print_final_summary
)


def main():
    """Main execution function."""
    print_header("NAIVE BAYES IRIS CLASSIFICATION PROJECT")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis project compares two implementations of Gaussian Naive Bayes:")
    print("  1. Custom implementation using NumPy and manual calculations")
    print("  2. Scikit-learn implementation using GaussianNB")

    # Step 1: Data Preparation
    print_header("STEP 1: DATA PREPARATION")
    data_path = os.path.join("..", "IRIS.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: Could not find IRIS.csv at {data_path}")
        sys.exit(1)

    data = prepare_data(data_path)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names, class_names = data['feature_names'], data['class_names']

    print("\n[OK] Data preparation completed successfully!")

    # Step 2: Exploratory Visualizations
    print_header("STEP 2: EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"[OK] Created plots directory: {plots_dir}")

    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    create_exploratory_visualizations(X_all, y_all, feature_names, class_names, plots_dir)

    # Steps 3-4: Train and Evaluate Models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test,
                                        feature_names, class_names)

    # Step 5: Compare Implementations
    print_header("STEP 5: COMPARING IMPLEMENTATIONS")
    pred_comparison = compare_predictions(results['custom']['predictions'],
                                         results['sklearn']['predictions'],
                                         y_test, class_names)

    compare_parameters(results['custom']['model'], results['sklearn']['model'],
                      feature_names, class_names)

    compare_metrics(results['custom']['metrics'], results['sklearn']['metrics'],
                   class_names)

    # Step 6: Generate Visualizations
    print_header("STEP 6: GENERATING COMPARISON VISUALIZATIONS")
    create_comparison_visualizations(results, class_names, feature_names,
                                    X_train, y_train, plots_dir)

    # Step 7: Generate Summary Report
    print_header("STEP 7: GENERATING SUMMARY REPORT")
    save_summary_report(results, class_names, pred_comparison)

    # Final Summary
    print_header("EXECUTION COMPLETED SUCCESSFULLY")
    print_final_summary(results, pred_comparison, plots_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
