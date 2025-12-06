# Product Requirements Document (PRD)
# Naive Bayes Classification on IRIS Dataset

## Project Overview
This project implements a comprehensive Naive Bayes classification exercise using the IRIS dataset. The implementation will compare two approaches: a custom implementation using NumPy with manual mathematical calculations, and the scikit-learn library implementation.

---

## 1. Project Objectives

### Primary Goals
- Build a Naive Bayes classifier from scratch using NumPy and mathematical calculations
- Implement Naive Bayes using scikit-learn library
- Compare both implementations and analyze differences
- Visualize data distributions and classification results
- Provide educational insights into Naive Bayes algorithm mechanics

### Learning Outcomes
- Understand the mathematical foundation of Naive Bayes
- Learn the difference between manual implementation and library usage
- Gain insights into probability distributions and Bayesian statistics
- Practice data visualization and model evaluation

---

## 2. Dataset Specifications

### Source Data
- **File**: IRIS.csv
- **Features**: 4 numerical features
  - sepal_length
  - sepal_width
  - petal_length
  - petal_width
- **Target**: species (3 classes)
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Data Split**: Pre-labeled in CSV
  - Training samples: unmarked rows
  - Test samples: rows marked with "TEST"

### Data Statistics
- Total samples: ~150
- Test samples: ~30 (marked as "TEST")
- Training samples: ~120 (unmarked)
- Features: 4 continuous numerical variables
- Classes: 3 balanced classes

---

## 3. Implementation Requirements

### 3.1 Custom NumPy Implementation

#### Core Components
1. **Data Preprocessing**
   - Load and parse IRIS.csv
   - Separate training and test sets based on "Training / Test" column
   - Convert species names to numerical labels
   - Handle missing values (if any)

2. **Probability Calculations**
   - Calculate prior probabilities P(class)
   - Calculate mean (μ) and standard deviation (σ) for each feature per class
   - Implement Gaussian probability density function:
     ```
     P(x|class) = (1 / √(2πσ²)) * exp(-(x-μ)²/(2σ²))
     ```

3. **Classification Logic**
   - For each test sample, calculate posterior probability for each class
   - Apply Naive Bayes formula:
     ```
     P(class|features) ∝ P(class) * ∏P(feature_i|class)
     ```
   - Predict class with highest posterior probability

4. **Model Evaluation**
   - Calculate accuracy
   - Generate confusion matrix
   - Calculate precision, recall, and F1-score per class

### 3.2 Scikit-learn Implementation

#### Core Components
1. **Data Preprocessing**
   - Same data loading as custom implementation
   - Ensure data format compatibility with sklearn

2. **Model Training**
   - Use `GaussianNB` from sklearn.naive_bayes
   - Fit model on training data
   - No hyperparameter tuning required (Naive Bayes is parameter-free)

3. **Model Evaluation**
   - Predict on test set
   - Calculate same metrics as custom implementation
   - Use sklearn's built-in evaluation functions

---

## 4. Visualization Requirements

### 4.1 Exploratory Data Analysis
1. **Feature Distribution Histograms**
   - Create histograms for each of the 4 features
   - Show distribution per class (3 classes overlaid or side-by-side)
   - Use different colors for each species
   - Include labels, legends, and titles in English

2. **Feature Relationships**
   - Scatter plots showing relationships between feature pairs
   - Color-coded by species
   - At least 2-3 key feature combinations

### 4.2 Model Performance Visualization
1. **Confusion Matrices**
   - Heatmap for custom NumPy implementation
   - Heatmap for scikit-learn implementation
   - Side-by-side comparison

2. **Accuracy Comparison**
   - Bar chart comparing both implementations
   - Show overall accuracy and per-class accuracy

3. **Probability Distributions**
   - Visualize learned Gaussian distributions (mean ± std)
   - Show for each feature and class combination

---

## 5. Comparison and Analysis

### Required Comparisons
1. **Accuracy Metrics**
   - Overall accuracy
   - Per-class precision, recall, F1-score
   - Confusion matrix analysis

2. **Implementation Differences**
   - Code complexity comparison
   - Execution time comparison
   - Memory usage (if measurable)

3. **Mathematical Insights**
   - Explain why results might differ (numerical precision, smoothing, etc.)
   - Discuss assumptions made by each implementation
   - Analyze edge cases or misclassifications

### Expected Differences Analysis
Document reasons for any differences:
- **Numerical precision**: Different floating-point calculations
- **Smoothing**: Sklearn may use Laplace smoothing by default
- **Variance handling**: How zero or near-zero variance is handled
- **Probability calculations**: Log-space vs. regular space computations

---

## 6. Technical Specifications

### Dependencies
```python
- numpy: Numerical computations and array operations
- pandas: Data loading and manipulation
- matplotlib: Plotting and visualization
- seaborn: Statistical visualizations
- scikit-learn: Machine learning implementation
```

### Code Structure
```
ai-course-exercise/
├── naive_bayes_custom.py      # Custom NumPy implementation
├── naive_bayes_sklearn.py     # Scikit-learn implementation
├── data_loader.py             # Data loading utilities
├── visualization.py           # All visualization functions
├── comparison.py              # Comparison and analysis
└── main.py                    # Main execution script
```

### Output Requirements
1. **Console Output**
   - Training progress indicators
   - Model parameters (means, standard deviations, priors)
   - Evaluation metrics for both implementations
   - Comparison summary

2. **Visual Output**
   - All plots saved as PNG files
   - High resolution (300 DPI minimum)
   - Clear labels in English
   - Professional styling

3. **Analysis Report**
   - Markdown or text file with findings
   - Explanation of differences
   - Insights and conclusions

---

## 7. Success Criteria

### Functional Requirements
- [x] Both implementations produce predictions
- [x] Accuracy > 85% on test set (both methods)
- [x] All required visualizations generated
- [x] Comprehensive comparison completed

### Quality Requirements
- [x] Code is well-documented with comments
- [x] Functions have docstrings
- [x] Variable names are descriptive
- [x] Code follows PEP 8 style guidelines
- [x] All text outputs in English

### Educational Requirements
- [x] Clear explanation of Naive Bayes mathematics
- [x] Step-by-step calculation walkthrough
- [x] Insights into probability distributions
- [x] Understanding of implementation tradeoffs

---

## 8. Deliverables

1. **Python Scripts** (in `ai-course-exercise/` folder)
   - All source code files as specified in section 6
   - Executable main.py that runs entire pipeline

2. **Visualizations**
   - Feature distribution histograms (4 plots)
   - Feature relationship scatter plots (3+ plots)
   - Confusion matrices (2 plots)
   - Accuracy comparison charts (1 plot)
   - Probability distribution visualizations (12+ plots)

3. **Documentation**
   - This PRD document
   - Tasks file (after PRD approval)
   - Code comments and docstrings
   - Analysis report with findings

4. **Results**
   - Classification accuracy metrics
   - Confusion matrices
   - Comparison analysis
   - Explanation of differences

---

## 9. Timeline and Milestones

### Phase 1: Setup and Data Preparation
- Create project structure
- Implement data loading functions
- Create initial visualizations

### Phase 2: Custom Implementation
- Implement Gaussian probability calculations
- Build Naive Bayes classifier from scratch
- Test and validate on training data

### Phase 3: Scikit-learn Implementation
- Implement sklearn version
- Ensure data compatibility
- Generate predictions

### Phase 4: Comparison and Visualization
- Create all required plots
- Calculate comparison metrics
- Generate analysis report

### Phase 5: Documentation and Finalization
- Complete code documentation
- Finalize visualizations
- Write comparison analysis
- Final review and testing

---

## 10. Assumptions and Constraints

### Assumptions
- Data is clean and properly formatted
- Gaussian distribution is appropriate for features
- Features are conditionally independent (Naive Bayes assumption)
- Test/Training split is representative

### Constraints
- Must use existing IRIS.csv file
- No external data sources
- Must implement both methods for comparison
- All outputs in English

---

## 11. Risk Assessment

### Potential Issues
1. **Numerical Instability**
   - Risk: Underflow in probability calculations
   - Mitigation: Use log-probabilities

2. **Zero Variance**
   - Risk: Division by zero in Gaussian calculation
   - Mitigation: Add small epsilon value

3. **Implementation Differences**
   - Risk: Results don't match between implementations
   - Mitigation: This is expected; document and explain

4. **Visualization Complexity**
   - Risk: Too many plots, unclear presentation
   - Mitigation: Use subplots and clear organization

---

## 12. Future Enhancements (Out of Scope)

- Cross-validation implementation
- Hyperparameter tuning (e.g., variance smoothing)
- Other Naive Bayes variants (Multinomial, Bernoulli)
- Feature selection analysis
- ROC curves and AUC scores
- Interactive visualizations

---

## Approval

This PRD requires approval before proceeding to implementation.

**Questions for Review:**
1. Are the visualization requirements sufficient?
2. Should we include additional performance metrics?
3. Is the code structure appropriate?
4. Should we add any specific analysis or comparisons?
5. Are there any additional educational insights needed?

---

**Document Version**: 1.0
**Last Updated**: 2025-12-06
**Status**: Pending Approval
