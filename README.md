# ML Preprocessing & Feature Selection — Personalized Diet Recommendation

## 1. Project Title
**Machine Learning Preprocessing and Feature Selection for Personalized Diet Recommendation**

## 2. Problem Statement
The project treats personalized diet recommendation as a **multi-class classification** problem. The goal is to clean demographic, physical, health, lifestyle and dietary information, select useful predictors, avoid data leakage, and prepare an ML-ready dataset for predicting `Recommended_Meal_Plan`.

## 3. Member
- Member : ALOK SINGH


## 4. Dataset Description
The supplied dataset contains **5,000 observations and 30 columns**. The target is `Recommended_Meal_Plan`, with four observed classes: Balanced Diet, High-Protein Diet, Low-Carb Diet and Low-Fat Diet.

The notebook identifies 19 numerical columns and 11 categorical/object columns before excluding the ID, target-derived recommendation outputs, and target itself from predictive candidates. fileciteturn0file0L487-L496

## 5. Dataset Source
**Source available to this repository:** the dataset supplied with the practical assignment / project materials. The uploaded notebook does not document an external URL, Kaggle page, DOI, or named publisher, so no external source is claimed here.

## 6. Preprocessing Techniques Implemented
- Missing-value audit
- Numerical missing-value treatment using training-set median
- Categorical missing-value treatment using `Unknown/Not Reported`
- Categorical whitespace normalization
- Exact duplicate detection and removal only if present
- Domain-plausibility diagnostics
- IQR and Z-score outlier detection
- Conservative outlier treatment: statistically extreme values are retained unless separately shown to be invalid
- Log1p transformation demonstration for a suitable non-negative feature
- Min-max normalization demonstration
- Standardization
- 80/20 train-test split
- Training-only preprocessing parameters to reduce leakage

The notebook explicitly states the training-only imputation/scaling approach and uses an 80/20 split. fileciteturn1file0L749-L784

## 7. Feature-Selection Techniques Implemented
1. Variance Threshold
2. Pearson Correlation for numerical redundancy
3. Chi-Square test for categorical feature-target association
4. ANOVA F-test for numerical feature-target differences
5. Mutual Information for categorical/discrete dependence

## 8. From-Scratch Implementations
The repository implements manual versions of:
- variance
- Pearson correlation
- Chi-Square statistic
- ANOVA F-statistic
- entropy
- mutual information
- train/test splitting
- categorical one-hot encoding logic
- numerical scaling logic

These mirror the notebook's assignment-oriented demonstrations. fileciteturn5file0L4-L13

## 9. Results
The notebook's final feature-selection result is:
- **Weight_kg**
- **BMI**
- **Protein_Intake**
- **Dietary_Habits**

The resulting selected encoded matrix contains **4,000 training rows × 7 encoded columns** and **1,000 testing rows × 7 encoded columns**. fileciteturn5file0L22-L26 fileciteturn5file1L44-L51

A Logistic Regression demonstration is included in `src/run_analysis.py` and the GUI. Its metrics are generated when the script is run rather than hard-coded. On the notebook's 80/20 random split (seed 42), the current demonstration gives 24.9% accuracy, macro-F1 0.2267 and weighted-F1 0.2266.

## 10. Selected Features
| Feature | Type | Selection logic |
|---|---|---|
| Weight_kg | Numerical | ANOVA evidence |
| BMI | Numerical | ANOVA evidence |
| Protein_Intake | Numerical | ANOVA evidence |
| Dietary_Habits | Categorical | Strongest MI feature under the notebook's final rule |

The notebook explicitly records these four selected original features. fileciteturn5file0L22-L26

## 11. Key Findings
- `Patient_ID` is excluded because it is an identifier.
- `Recommended_Calories`, `Recommended_Protein`, `Recommended_Carbs`, and `Recommended_Fats` are treated as potential target-derived variables and excluded from prediction to reduce leakage risk. fileciteturn0file0L477-L496
- Exact duplicate count in the supplied data is zero.
- Missing values exist in the supplied raw data; the final transformation is designed to produce a complete numerical matrix.
- The final selected feature set is compact: 4 original predictors instead of the full set of predictive candidates.
- `Dietary_Habits` is retained because it is the strongest categorical Mutual Information feature in the notebook's final rule.

## 12. Instructions to Run the Code
### Google Colab
1. Open `notebooks/main_analysis.ipynb`.
2. Upload `dataset/raw_dataset.csv` when prompted, or change the dataset path.
3. Run cells from top to bottom.

### Local Python
Install:
```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

Run:
```bash
python src/run_analysis.py
```

Launch the desktop GUI:
```bash
python src/gui_app.py
```

The GUI shows dataset information, a preview, feature-selection statistics, and an interactive meal-plan prediction demonstration.

## 13. Google Colab Link
**Paste your final public Colab URL here:**  
`https://colab.research.google.com/drive/PASTE-YOUR-NOTEBOOK-ID-HERE`

## Repository Structure
```text
ML-Preprocessing-Feature-Selection/
├── README.md
├── dataset/
│   ├── raw_dataset.csv
│   ├── cleaned_dataset.csv
│   └── selected_features_dataset.csv
├── notebooks/
│   └── main_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── run_analysis.py
│   └── gui_app.py
├── results/
│   ├── graphs/
│   └── outputs/
└── report/
    └── final_summary.pdf
```
