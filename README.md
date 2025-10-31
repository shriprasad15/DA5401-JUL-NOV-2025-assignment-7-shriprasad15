# DA5401-JUL-NOV-2025-assignment-7-shriprasad15

# Multi-Class Model Selection using ROC and Precision-Recall Curves
### An in-depth analysis of classifier performance on the Landsat Satellite Dataset

- **Name:** S SHRIPRASAD
- **Roll:** DA25E054

This repository contains a comprehensive analysis for a multi-class classification problem. The primary goal is to select the best model for classifying land cover types from satellite imagery, moving beyond simple accuracy to conduct a deep-dive evaluation using Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.

---


## Project Overview

In machine learning, selecting the "best" model is rarely about picking the one with the highest accuracy. Real-world problems require a nuanced understanding of a model's behavior across different decision thresholds, its performance on imbalanced classes, and the reliability of its probability estimates.

This project tackles this challenge head-on. Using the classic UCI Landsat Satellite dataset, we compare a diverse squadron of classifiers—ranging from simple baselines and poorly performing models to powerful ensemble methods. The core of the analysis lies in the application and interpretation of **One-vs-Rest (OvR) ROC and Precision-Recall curves**, which provide a profound, threshold-independent view of model performance in a multi-class setting.

The narrative of this analysis follows a structured process:
1.  Establish a baseline performance with simple metrics.
2.  Conduct a deep-dive ROC and PR analysis to rank models robustly.
3.  Perform per-class diagnostics to identify specific weaknesses.
4.  Investigate advanced topics like probability calibration and feature importance.
5.  Provide actionable recommendations for future work.

## Problem Statement

As a machine learning scientist, the task is to classify land cover types using satellite image data from the UCI Landsat Satellite dataset. This is a non-trivial multi-class problem (6 classes) characterized by high feature dimensionality and potential class overlap.

The primary goal is to perform rigorous model selection. Instead of relying on single-point metrics like accuracy, the analysis must leverage **ROC curves** and **Precision-Recall curves**, adapted for the multi-class setting, to determine the best- and worst-performing models. Special attention is paid to interpreting model behavior across the full spectrum of decision thresholds.

## Dataset

*   **Name:** Landsat Satellite Dataset (Statlog version)
*   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)
*   **Characteristics:**
    *   Multi-class classification problem with 6 primary classes (e.g., red soil, grey soil, etc.).
    *   36 numeric features representing spectral values of pixels in a 3x3 neighborhood.
    *   Training set: 4,435 samples.
    *   Testing set: 2,000 samples.
*   **Citation:** Blake, C. and Merz, C.J. (1998). *UCI Repository of machine learning databases*. Irvine, CA: University of California, Department of Information and Computer Science.

## Methodology

The project follows a systematic workflow to ensure a robust and reproducible analysis.

1.  **Data Preparation & Preprocessing:**
    *   The data was loaded directly from the UCI repository.
    *   Features were standardized using `StandardScaler` to ensure that models sensitive to feature scale (like SVC and Logistic Regression) could perform optimally.
    *   Target labels, which were originally non-sequential (`[1, 2, 3, 4, 5, 7]`), were encoded into a zero-indexed format (`[0, 1, 2, 3, 4, 5]`) using `LabelEncoder` for compatibility with all libraries, especially XGBoost.

2.  **Model Training & Baseline Evaluation:**
    *   A diverse set of eight models was trained to cover a wide performance spectrum: `DummyClassifier` (as a sanity check), `GaussianNB`, `DecisionTree`, `LogisticRegression`, `KNeighborsClassifier`, `SVC`, and two powerful ensembles: `RandomForest` and `XGBoost`.
    *   Initial performance was gauged using Overall Accuracy and Weighted F1-Score.

3.  **Advanced Evaluation (ROC & PR Analysis):**
    *   Since this is a multi-class problem, a **One-vs-Rest (OvR)** strategy was adopted. For each model, six distinct ROC and PR curves were generated (one for each class against all others).
    *   These individual curves were then aggregated using **macro-averaging** to produce a single, summary ROC curve and PR curve for each model. This approach treats every class equally, providing a balanced view of overall performance.
    *   The **Area Under the ROC Curve (Macro-AUC)** and **Average Precision (Macro-AP)** were used as the primary metrics for model ranking.

4.  **Conceptual Demonstrations:**
    *   A custom `ContrarianClassifier` was engineered to demonstrate the concept of an anti-predictor—a model with a Macro-AUC significantly less than 0.5. This was achieved by systematically inverting the probability outputs of a well-trained base model.

5.  **Per-Class Diagnostics & Recommendations:**
    *   After identifying the champion model (`RandomForest`), a deep-dive analysis was performed on it, including plotting per-class ROC/PR curves, inspecting the confusion matrix, analyzing probability calibration, and examining feature importances.


## Technology Stack

*   **Python 3.10+**
*   **Core Libraries:**
    *   `pandas` & `numpy` for data manipulation
    *   `scikit-learn` for modeling, preprocessing, and metrics
    *   `matplotlib` & `seaborn` for visualization
    *   `xgboost` for the XGBoost classifier
*   **Environment:** Jupyter Notebook

To install all necessary dependencies, run:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter
```

## How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shriprasad15/DA5401-JUL-NOV-2025-assignment-7-shriprasad15
    cd DA5401-JUL-NOV-2025-assignment-7-shriprasad15
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Launch Jupyter Notebook and run the analysis:**
    ```bash
    jupyter notebook DA5401_A7_Analysis.ipynb
    ```

## Key Learnings & Recommendations

*   **Final Recommendation:** The **Random Forest** classifier is the recommended model for this task. It achieved the highest Macro-AUC (0.993) and Macro-AP (0.925), demonstrated stable performance under cross-validation, and its probabilities were reasonably well-calibrated.

*   **Actionable Next Steps:**
    1.  **Address Class Confusion:** Focus on the specific misclassifications identified in the confusion matrix (e.g., between classes 3, 4, and 7). This could involve collecting more data for these classes or engineering features that better separate them.
    2.  **Probability Calibration:** For applications requiring high-confidence probability scores (e.g., risk assessment), the Random Forest model should be formally calibrated using `CalibratedClassifierCV` before deployment.
    3.  **Feature Selection:** The feature importance analysis revealed that a subset of the 36 features drives most of the predictive power. Experimenting with a simpler model using only the top 15-20 features could lead to a faster, more efficient model with a negligible drop in performance.
    4.  **Hyperparameter Tuning:** While the default models performed exceptionally well, a systematic hyperparameter tuning process (e.g., using `GridSearchCV` or `RandomizedSearchCV`) on the top models (Random Forest, XGBoost, SVC) would likely yield further performance gains.
