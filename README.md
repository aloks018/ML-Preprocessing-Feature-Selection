# Machine Learning Preprocessing & Feature Selection for Personalized Diet Recommendation

A machine learning project focused on **data preprocessing, statistical feature selection, and preparation of a reliable dataset for personalized meal-plan prediction**.

---

## Student Details

| Particular          | Details                                                  |
| ------------------- | -------------------------------------------------------- |
| **Name**            | **Alok Singh**                                           |
| **Course**          | **MSc Mathematics with AI and Data Science**             |
| **Roll No.**        | **25225100002**                                          |
| **Enrollment No.**  | **CSJMA25000006137**                                     |
| **Project Type**    | Individual / Single Member                               |
| **Project Domain**  | Machine Learning, Data Preprocessing & Feature Selection |
| **Target Variable** | `Recommended_Meal_Plan`                                  |

---

## Project Overview

The objective of this project is to develop a structured machine-learning workflow for a **Personalized Diet Recommendation** problem.

The supplied dataset contains patient-related demographic, physical, health, lifestyle, and dietary information. The project prepares this information for a machine-learning classification task in which the model predicts the patient's:

**`Recommended_Meal_Plan`**

The work emphasizes two important stages before model development:

1. **Data Preprocessing** – improving data quality and converting the raw dataset into a machine-learning-compatible representation.
2. **Feature Selection** – identifying informative predictors and removing redundant or potentially leakage-prone variables.

The repository is therefore designed as a clean foundation for the subsequent development of a complete personalized diet prediction system.

---

## Problem Statement

Personalized diet recommendation can be formulated as a **multi-class classification problem**.

Given relevant patient characteristics, the objective is to predict an appropriate meal-plan category.

### Target

`Recommended_Meal_Plan`

### Observed Meal-Plan Classes

* Balanced Diet
* High-Protein Diet
* Low-Carb Diet
* Low-Fat Diet

The project intentionally separates useful predictive information from identifiers and variables that could introduce target leakage.

---

## Dataset

The supplied dataset contains:

* **5,000 observations**
* **30 columns**
* **19 numerical variables**
* **11 categorical/object variables**

The target variable is:

```text
Recommended_Meal_Plan
```

The original dataset also contains recommendation-related fields such as:

```text
Recommended_Calories
Recommended_Protein
Recommended_Carbs
Recommended_Fats
```

These fields are excluded from the predictive feature set because they may contain information derived from the recommendation process itself.

`Patient_ID` is also excluded because it is an identifier rather than a meaningful predictive attribute.

---

## Dataset Source

The dataset used in this project is the dataset supplied with the practical/project materials.

No external Kaggle, DOI, publisher, or third-party dataset URL is claimed because no such source is documented in the project materials.

---

## Project Workflow

The project follows this overall pipeline:

```text
Raw Dataset
     │
     ▼
Data Inspection & Quality Analysis
     │
     ▼
Missing-Value Analysis
     │
     ▼
Data Cleaning & Normalization
     │
     ▼
Duplicate / Outlier Diagnostics
     │
     ▼
Train-Test Split
     │
     ▼
Feature Selection
     │
     ├── Variance Analysis
     ├── Pearson Correlation
     ├── ANOVA
     ├── Chi-Square
     └── Mutual Information
     │
     ▼
Final Selected Features
     │
     ▼
Encoding + Imputation + Scaling
     │
     ▼
Machine-Learning Model Demonstration
     │
     ▼
Recommended Meal Plan
```

---

# 1. Data Preprocessing

The preprocessing component is implemented in `preprocessing.py`.

Its purpose is to transform raw patient data into a form that can safely be used for machine learning.

### Main preprocessing operations

* Dataset loading using Pandas
* Missing-value handling
* Numerical imputation using the median
* Categorical missing-value handling
* Category text normalization
* Categorical encoding
* Removal of patient identifiers
* Removal of target-derived recommendation fields
* Preparation of predictor matrix `X`
* Preparation of target vector `y`

The preprocessing module defines:

```python
TARGET = "Recommended_Meal_Plan"
ID_COL = "Patient_ID"
```

and excludes:

```text
Patient_ID
Recommended_Calories
Recommended_Protein
Recommended_Carbs
Recommended_Fats
```

from the predictive feature candidates.

### Final preprocessing concept

```text
Raw Dataset
   ↓
Clean / Normalize
   ↓
Remove ID & Leakage Columns
   ↓
X = Predictor Variables
y = Recommended_Meal_Plan
```

---

# 2. Feature Selection

The feature-selection component is implemented in:

```text
feature_selection.py
```

This module contains assignment-oriented implementations of several statistical techniques rather than treating feature selection as a black-box operation.

## Techniques Used

### Variance

Variance is used to understand whether a numerical feature contains meaningful variation.

A near-zero-variance feature provides little discriminatory information.

### Pearson Correlation

Pearson correlation is used to identify strong relationships between numerical variables.

The project treats a correlation magnitude of:

```text
|r| >= 0.85
```

as a redundancy warning.

When two numerical features are highly correlated, the feature with stronger ANOVA evidence is retained.

### ANOVA

ANOVA is applied to numerical features to determine whether the feature differs significantly across meal-plan classes.

The final selection rule uses:

```text
ANOVA p-value < 0.05
```

for numerical feature selection.

### Chi-Square Test

Chi-square is used to evaluate the association between categorical features and:

```text
Recommended_Meal_Plan
```

Categorical features with:

```text
Chi-square p-value < 0.05
```

are selected.

### Mutual Information

Mutual Information measures the amount of information a categorical feature provides about the target.

The feature with the strongest categorical MI is retained under the project's final selection rule, even when necessary to supplement the Chi-square selection.

The feature-selection module implements these calculations directly in Python, including variance, Pearson correlation, Chi-square, ANOVA, entropy, and mutual information.

---

# 3. Final Selected Features

The final feature-selection stage identifies the following original features:

| Feature          | Type        | Role in Selection                                |
| ---------------- | ----------- | ------------------------------------------------ |
| `Weight_kg`      | Numerical   | Selected through ANOVA evidence                  |
| `BMI`            | Numerical   | Selected through ANOVA evidence                  |
| `Protein_Intake` | Numerical   | Selected through ANOVA evidence                  |
| `Dietary_Habits` | Categorical | Retained as the strongest categorical MI feature |

Therefore, the final predictive feature set is:

```text
Weight_kg
BMI
Protein_Intake
Dietary_Habits
```

The repository's selected dataset contains these predictors together with the target variable `Recommended_Meal_Plan`.

---

# 4. Why Feature Selection Is Important

Using every available column is not always a good machine-learning strategy.

Feature selection helps the project to:

* Remove irrelevant predictors
* Reduce redundancy between highly correlated variables
* Reduce dimensionality
* Improve interpretability
* Reduce the risk of data leakage
* Provide the model with a more meaningful input space

For this project, the final selected set is intentionally compact rather than using all original predictive candidates.

---

# 5. Train-Test Strategy

The implementation uses an **80/20 train-test split**.

```text
80% → Training Data
20% → Testing Data
```

A fixed random seed of:

```text
42
```

is used to make the split reproducible.

The training portion is used for feature-selection decisions, while the test portion is retained for evaluation. The execution script creates the split first and then performs feature selection using the training data.

---

# 6. Machine-Learning Demonstration

After feature selection, the selected numerical and categorical predictors are passed through a preprocessing pipeline.

### Numerical features

* Median imputation
* Standardization

### Categorical features

* Most-frequent-value imputation
* One-hot encoding
* Unknown categories handled safely

### Demonstration model

The project includes a **Logistic Regression** classifier.

The complete execution script trains the model on the selected feature set and generates:

* Accuracy
* Macro F1-score
* Weighted F1-score
* Classification report

The metrics are generated by the script rather than being manually hard-coded.

---

# 7. Project Outputs

The project produces several useful artifacts.

### Dataset files

```text
raw_dataset.csv
cleaned_dataset.csv
selected_features_dataset.csv
```

### Python modules

```text
preprocessing.py
feature_selection.py
run_analysis.py
```

### Notebook

```text
main_analysis.ipynb
```

### Visualizations

```text
class_distribution.png
anova_pvalues.png
```

### Analysis outputs

```text
anova_results.csv
categorical_selection.csv
selected_features.csv
metrics.json
classification_report.txt
```

### Report

```text
final_summary.pdf
```

These files provide both the reproducible analysis workflow and the intermediate/final outputs of the project. The current repository root includes the notebook, preprocessing and feature-selection modules, datasets, plots, execution script, and PDF summary.

---

# 8. Repository Structure

The current repository is organized around the following project files:

```text
ML-Preprocessing-Feature-Selection/
│
├── README.md
├── raw_dataset.csv
├── cleaned_dataset.csv
├── selected_features_dataset.csv
│
├── main_analysis.ipynb
│
├── preprocessing.py
├── feature_selection.py
├── run_analysis.py
│
├── anova_pvalues.png
├── class_distribution.png
│
└── final_summary.pdf
```

---

# 9. File-wise Role

| File                            | Purpose                                                                            |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| `raw_dataset.csv`               | Original supplied dataset                                                          |
| `cleaned_dataset.csv`           | Dataset after preprocessing/cleaning                                               |
| `selected_features_dataset.csv` | Dataset containing the final selected predictors and target                        |
| `main_analysis.ipynb`           | Main analysis and assignment workflow                                              |
| `preprocessing.py`              | Reusable preprocessing functions                                                   |
| `feature_selection.py`          | Statistical feature-selection functions                                            |
| `run_analysis.py`               | Executes the end-to-end preprocessing, selection, training and evaluation workflow |
| `anova_pvalues.png`             | Visualization of numerical feature ANOVA evidence                                  |
| `class_distribution.png`        | Distribution of meal-plan target classes                                           |
| `final_summary.pdf`             | Project summary/report                                                             |

---

# 10. Reproducibility

The project is designed to make the analysis reproducible.

Important reproducibility choices include:

* Fixed random seed: `42`
* Explicit 80/20 train-test split
* Defined target column
* Explicit leakage-column exclusion
* Deterministic feature-selection rules
* Pipeline-based imputation, encoding and scaling
* Programmatically generated evaluation metrics

---

# 11. How to Run the Project

## Option 1 — Google Colab

Open the notebook:

**[Open Google Colab Notebook](https://colab.research.google.com/drive/1xCTkheKnvXF7S1C3v95MANRvb4mI-mrg?usp=sharing)**

Run the notebook cells in sequence.

---

## Option 2 — Run Python Locally

Install the required libraries:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

Then run:

```bash
python run_analysis.py
```

The execution script performs the main analysis, applies feature selection, trains the Logistic Regression demonstration model, calculates evaluation metrics, and saves analysis artifacts.

---

# 12. Key Project Findings

The analysis leads to several important conclusions:

1. `Patient_ID` is an identifier and is not used as a predictive feature.
2. Recommendation-derived columns are excluded to reduce the risk of target leakage.
3. Statistical feature-selection methods are used instead of selecting variables arbitrarily.
4. Strongly correlated numerical variables are checked for redundancy.
5. The final feature set is reduced to four meaningful original predictors.
6. `Dietary_Habits` is retained through the categorical information-selection rule.
7. The final selected dataset is substantially smaller and more interpretable than the original feature space.

---

# 13. Important Project Limitation

This repository primarily focuses on:

**Preprocessing + Feature Selection + ML-ready Dataset Preparation + Model Demonstration**

It should therefore be viewed as a preprocessing and analytical foundation for a personalized diet recommendation system rather than as a clinically validated dietary decision system.

Predictions from the machine-learning component should not be interpreted as medical advice.

---

# 14. Project Links

### GitHub Repository

**https://github.com/aloks018/ML-Preprocessing-Feature-Selection**

### Google Colab Notebook

**https://colab.research.google.com/drive/1xCTkheKnvXF7S1C3v95MANRvb4mI-mrg?usp=sharing**

### Raw Dataset

**https://github.com/aloks018/ML-Preprocessing-Feature-Selection/blob/main/raw_dataset.csv**

---

# 15. Author

**Alok Singh**

**MSc Mathematics with AI and Data Science**

**Roll No.: 25225100002**

**Enrollment No.: CSJMA25000006137**

This is an **individual project completed by one member**.


## Conclusion

This project demonstrates a systematic machine-learning data preparation workflow for personalized diet recommendation.

The main contribution is not simply preparing a dataset, but establishing a logical sequence in which:

Data Quality
     ↓
Preprocessing
     ↓
Leakage Control
     ↓
Statistical Feature Analysis
     ↓
Feature Selection
     ↓
Encoding / Scaling
     ↓
Machine-Learning Ready Data
     ↓
Meal-Plan Prediction

The final selected predictors — `Weight_kg`, `BMI`, `Protein_Intake`, and `Dietary_Habits` — provide a compact and interpretable feature space for the downstream personalized meal-plan classification task.
