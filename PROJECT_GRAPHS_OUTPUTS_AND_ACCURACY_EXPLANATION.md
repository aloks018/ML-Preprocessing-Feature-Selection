#  Graphs, Outputs and Accuracy 

## Project

**Machine Learning Preprocessing and Feature Selection for Personalized
Diet Recommendation**

This document explains the important graphs, tables, statistical
outputs, preprocessing results, feature-selection decisions, and
model-accuracy status from the uploaded `main_analysis.ipynb`.


------------------------------------------------------------------------

# 1. Dataset Overview

The notebook works with a dataset containing:

-   **5,000 records/observations**
-   **30 columns**
-   **Target variable:** `Recommended_Meal_Plan`
-   **Target classes:** Balanced Diet, High-Protein Diet, Low-Carb Diet,
    Low-Fat Diet
-   **Numerical variables:** 19
-   **Categorical/object variables:** 11

The target is a categorical classification variable because the output
is one of four meal-plan categories.


# 2. First Five / Last Five Records

The notebook displays the first and last five records.

### Why this is done

It is a basic sanity check. It helps confirm:

-   the columns are loaded correctly;
-   values look reasonable;
-   categorical and numerical data are present;
-   the target column is available;
-   missing values can be visually noticed.

### Important observation

The dataset contains fields such as:

-   Age
-   Gender
-   Height
-   Weight
-   BMI
-   Blood pressure
-   Cholesterol
-   Blood sugar
-   Lifestyle variables
-   Dietary variables
-   Current calorie/protein/carbohydrate/fat intake
-   Recommended nutrition values
-   Recommended meal plan


# 3. Missing-Value Analysis

The notebook calculates missing values and missing percentages for every
column.

The important missing values are:

  Feature             Missing Values   Missing %
  ----------------- ---------------- -----------
  Chronic_Disease              2,043      40.86%
  Allergies                    3,497      69.94%
  Food_Aversions               1,225      24.50%

The numerical features do not have missing values in the raw dataset.

### What does Missing % mean?

The formula used is:

`Missing Percentage = (Missing Values / Total Observations) × 100`

For example:

`2043 / 5000 × 100 = 40.86%`

### Treatment used

For numerical variables:

> Missing values are handled using the **training-set median**.

For categorical variables:

> Missing values are represented as **`Unknown/Not Reported`**.

This is particularly appropriate here because replacing an unknown
categorical response with the most common category could create
misleading information.

### Very important ML point

The final preprocessing parameters are learned from the **training data
only**.

This prevents **data leakage**.

### Output

> "I did not blindly delete rows containing missing values. I used
> median imputation for numerical variables and an Unknown/Not Reported
> category for categorical variables. The final preprocessing parameters
> are learned only from the training set to avoid leakage."

------------------------------------------------------------------------

# 4. Duplicate Analysis

The notebook performs duplicate detection in two ways:

-   Manual duplicate count
-   Pandas verification

Both give:

**Duplicate records = 0**

Therefore:

-   Original records = 5,000
-   Duplicate records = 0
-   Records after duplicate treatment = 5,000

### Why duplicates matter

If exact duplicate rows are present, the same observation can receive
extra influence during analysis or model training.

### Output

> "I checked exact duplicate records before modeling. The dataset
> contains zero exact duplicates, so no records were removed."

------------------------------------------------------------------------

# 5. Invalid and Inconsistent Data Check

The notebook checks:

-   extra spaces;
-   capitalization/case differences;
-   possible categorical variants;
-   impossible or suspicious numerical values.

The categorical checks found:

**No extra outer spaces** and **no detected case/space variants** in the
listed categorical variables.

The numerical plausibility check also found:

**0 potentially invalid values** for the diagnostic bounds used in the
notebook.

### Important wording

These numerical limits are described in the notebook as a **diagnostic
plausibility screen**, not as official medical limits.

### Output

> "I used the invalid-data checks as diagnostic screening. I did not
> automatically delete extreme values just because they looked unusual."

------------------------------------------------------------------------

# 6. Label Encoding

The notebook demonstrates label encoding using `Genetic_Risk_Factor`.

Mapping:

-   `No → 0`
-   `Yes → 1`

### What is Label Encoding?

It converts categories into numbers.

Example:

`No = 0`

`Yes = 1`

### Why is it suitable here?

It is reasonable for a binary variable.

### Limitation

For nominal categories with no natural order, assigning values such as:

`Asian = 0, Indian = 1, Western = 2`

can incorrectly suggest that one category is mathematically greater than
another.

Therefore, the notebook uses **one-hot encoding for nominal categorical
variables in the final predictive matrix**.

------------------------------------------------------------------------

# 7. One-Hot Encoding

The notebook demonstrates one-hot encoding using `Preferred_Cuisine`.

The categories are:

-   Asian
-   Indian
-   Mediterranean
-   Western

The output creates four binary columns:

-   `Preferred_Cuisine_Asian`
-   `Preferred_Cuisine_Indian`
-   `Preferred_Cuisine_Mediterranean`
-   `Preferred_Cuisine_Western`

### Meaning of 0 and 1

For one observation:

-   `1` means the observation belongs to that category.
-   `0` means it does not.

### Why use one-hot encoding?

It avoids creating an artificial ranking between nominal categories.

### Output

> "I used one-hot encoding for nominal categorical variables because
> categories such as cuisine types do not have a natural numerical
> order."

------------------------------------------------------------------------

# 8. Outlier Detection -- IQR Method

The notebook uses the Interquartile Range method.

### Formula

`IQR = Q3 − Q1`

`Lower Bound = Q1 − 1.5 × IQR`

`Upper Bound = Q3 + 1.5 × IQR`

An observation outside these bounds is flagged as a potential outlier.

### Important result

The IQR analysis detected:

**4 potential outliers in BMI**

All other listed numerical features had zero IQR outlier flags.

### What does this mean?

It does **not** automatically mean those four BMI values are wrong.

They are statistically unusual according to the IQR rule.

### Treatment

The four BMI observations were **retained after plausibility checking**.

### Why retain them?

A real patient can legitimately have an unusually high or low BMI.
Removing such observations without evidence could remove useful
information.

### Output

> "The IQR method flagged four BMI observations. I did not automatically
> delete them because an outlier is not necessarily an error. I retained
> them after the plausibility check."

------------------------------------------------------------------------

# 9. Outlier Detection -- Z-Score Method

The notebook also uses:

`Z = (X − μ) / σ`

A common extreme-value rule is:

`|Z| > 3`

### Result

The Z-score method detected:

**0 observations with \|Z\| \> 3**

for the listed numerical variables.

### Why did IQR and Z-score differ?

The IQR method uses quartiles and is relatively robust to extreme
values.

The Z-score method uses the mean and standard deviation.

Therefore, they can produce different results.

### Output

> "I used both IQR and Z-score methods because they identify unusual
> observations using different statistical principles. In my dataset,
> IQR flagged four BMI values, while the \|Z\| \> 3 rule flagged none."

------------------------------------------------------------------------

# 10. Skewness and Log Transformation

The notebook checks skewness to see whether a variable is asymmetric.

### BMI result

Original BMI skewness:

**0.4265**

After `log1p` transformation:

**-0.1447**

The notebook marks this as an improvement in absolute skewness.

### What does skewness mean?

Skewness describes the asymmetry of a distribution.

-   Positive skewness → longer/right tail
-   Negative skewness → longer/left tail
-   Near zero → more symmetric

### Important point

The notebook treats transformation as something to investigate where
appropriate. It does not blindly apply a log transformation to every
variable.

### Output

> "I checked skewness before transformation. BMI had positive skewness
> of 0.4265, and the log1p demonstration reduced the absolute skewness
> to 0.1447."

------------------------------------------------------------------------

# 11. Feature Scaling -- Min-Max Normalization

The notebook demonstrates Min-Max normalization.

### Formula

`X_scaled = (X − X_min) / (X_max − X_min)`

The output range becomes:

**0 to 1**

For the demonstrated Age values:

-   Minimum = 18
-   Maximum = 79

Example:

Age = 56

Normalized value:

**0.622951**

### Meaning

A value near:

-   `0` → near the minimum
-   `1` → near the maximum

### Output

> "Min-Max normalization converts numerical values to a common 0-to-1
> scale."

------------------------------------------------------------------------

# 12. Feature Scaling -- Standardization

The notebook also demonstrates standardization.

### Formula

`Z = (X − μ) / σ`

After standardization:

-   Mean is approximately **0**
-   Population standard deviation is **1**

The notebook output confirms:

**Mean ≈ 9.24 × 10⁻¹⁷**

which is effectively zero.

**Population SD = 1.0**

### Meaning

A standardized value tells us how far an observation is from the mean in
standard-deviation units.

For example:

-   Positive value → above the mean
-   Negative value → below the mean
-   Near zero → close to the mean

### Output

> "I used standardization so numerical variables with different units
> and scales can be compared more fairly by transforming them to
> approximately mean zero and standard deviation one."

------------------------------------------------------------------------

# 13. Graph 1 -- BMI Histogram

### Graph title

**Distribution of BMI**

### X-axis

**BMI**

This represents BMI values.

### Y-axis

**Frequency**

This represents how many observations fall within each BMI interval.

### What is a histogram?

A histogram groups continuous numerical values into intervals called
bins.

Each bar represents the number of observations in one interval.

### What should I look for?

A histogram helps identify:

-   concentration;
-   spread;
-   possible skewness;
-   unusual distribution shape.

### Output

> "The BMI histogram shows how the 5,000 observations are distributed
> across BMI ranges. The x-axis contains BMI intervals and the y-axis
> shows the number of observations in each interval."

### Important distinction

A histogram is different from a bar chart:

-   Histogram → numerical continuous ranges
-   Bar chart → separate categories

------------------------------------------------------------------------

# 14. Graph 2 -- Blood Sugar Box Plot

### Graph title

**Box Plot of Blood Sugar Level**

### X-axis

**Blood Sugar Level**

### Y-axis

**Value**

### What does a box plot show?

A box plot summarizes a distribution using:

-   minimum/low end;
-   first quartile (Q1);
-   median;
-   third quartile (Q3);
-   upper end;
-   possible extreme observations.

The box represents the middle 50% of the observations.

### Why is this graph useful?

It visually complements the IQR outlier analysis.

### Output

> "The blood-sugar box plot gives a compact summary of the distribution.
> It helps me see the median, quartile spread and possible extreme
> observations."

------------------------------------------------------------------------

# 15. Graph 3 -- Recommended Meal Plan Bar Chart

### Graph title

**Recommended Meal Plan Class Distribution**

### X-axis

**Recommended Meal Plan**

The x-axis contains the four target categories:

-   Balanced Diet
-   High-Protein Diet
-   Low-Carb Diet
-   Low-Fat Diet

### Y-axis

**Number of Observations**

The height of each bar represents how many records belong to that class.

### Why is this graph important?

It checks whether the target classes are reasonably distributed.

If one class were much larger than the others, the dataset could be
class-imbalanced.

### Output

> "This bar chart shows the frequency of each meal-plan class. I use it
> to understand the distribution of the classification target and to
> check whether one class dominates the dataset."

### Important caution

The notebook's displayed interpretation says to assess whether the
target is roughly balanced or dominated by one class. Do not claim
perfect balance unless the actual bar heights support that statement.

------------------------------------------------------------------------

# 16. Graph 4 -- Weight vs BMI Scatter Plot

### Graph title

**Weight vs BMI**

### X-axis

**Weight (kg)**

### Y-axis

**BMI**

### What does each point represent?

Each dot represents one observation.

Therefore, the graph contains information from the patient records.

### What does the pattern mean?

If points show an upward pattern:

> Higher weight tends to be associated with higher BMI.

If the points are widely scattered:

> Weight alone does not completely determine BMI.

### Output

> "The scatter plot is used to visually inspect the relationship between
> weight and BMI. Each point is an observation, and the pattern tells us
> whether there is an apparent association."

### Important caution

Association does not automatically mean causation.

------------------------------------------------------------------------

# 17. Graph 5 -- Pearson Correlation Heatmap

### Graph title

**Pearson Correlation Heatmap -- Candidate Numerical Features**

The heatmap is based on the numerical candidate features.

### What is Pearson correlation?

Pearson correlation measures the strength and direction of a **linear
relationship** between two numerical variables.

Its range is:

`-1 ≤ r ≤ +1`

### Meaning of values

-   `r` close to `+1` → strong positive linear relationship
-   `r` close to `-1` → strong negative linear relationship
-   `r` close to `0` → weak linear relationship

### How to read the heatmap

Each cell represents the Pearson correlation between the feature on the
row and the feature on the column.

The color intensity represents the strength/direction of the correlation
according to the heatmap scale.

### Why is the heatmap important?

Highly correlated predictor variables may contain redundant information.

The notebook uses a threshold of:

**\|r\| ≥ 0.85**

to identify strongly correlated numerical feature pairs for redundancy
analysis.

### Important point

Correlation is used here mainly to investigate **relationships between
predictors**, not as the primary feature-target test, because the target
is categorical.

### Output

> "I use Pearson correlation to identify linear relationships and
> possible redundancy among numerical predictors. A high absolute
> correlation can indicate that two predictors carry similar
> information."

------------------------------------------------------------------------

# 18. Variance Threshold Analysis

Variance measures how much a numerical feature changes across
observations.

### Result

All 15 predictive numerical features had non-zero variance.

Therefore:

**No numerical feature was removed by zero-variance screening.**

### Important point

A feature having variance does not automatically mean it is useful for
predicting the target.

Therefore, the notebook keeps varying features for further statistical
analysis.

### Output

> "Variance screening is mainly a first filter. Since the predictive
> numerical features all varied, I did not remove them solely on the
> basis of variance."

------------------------------------------------------------------------

# 19. Pearson Correlation -- Feature Redundancy

The notebook calculates pairwise Pearson correlations between numerical
candidate features.

The purpose is:

**Find highly correlated predictors that may be redundant.**

The notebook defines a strong-correlation threshold of:

**\|r\| ≥ 0.85**

If two features are highly correlated, the final feature-selection logic
can remove the weaker one according to ANOVA evidence.

### Output

> "When two numerical predictors are highly correlated, I avoid keeping
> both automatically. I compare their ANOVA evidence and retain the
> stronger feature where redundancy is identified."

------------------------------------------------------------------------

# 20. Chi-Square Feature Selection

Chi-square is used for:

**Categorical feature vs categorical target**

The target `Recommended_Meal_Plan` has four categories.

### Main hypothesis

The test investigates whether the categorical feature and target appear
independent.

A small p-value gives evidence against independence.

### Results

  Feature                 Chi-Square   df    p-value
  --------------------- ------------ ---- ----------
  Dietary_Habits           16.104390    9   0.064733
  Genetic_Risk_Factor       5.519918    3   0.137452
  Food_Aversions           12.530620    9   0.185020
  Allergies                10.221066    9   0.332887
  Preferred_Cuisine         9.117588    9   0.426491
  Smoking_Habit             2.440681    3   0.486107
  Gender                    4.255137    6   0.642193
  Chronic_Disease           7.623231   12   0.813837
  Alcohol_Consumption       0.685563    3   0.876594

### Most important result

`Dietary_Habits` has the smallest categorical p-value:

**0.064733**

However:

**0.064733 \> 0.05**

So it does not pass the strict p \< 0.05 criterion.

The final selection still retains `Dietary_Habits` because the notebook
explicitly adds the strongest categorical Mutual Information feature.

### Output

> "Dietary_Habits had the strongest Chi-square evidence among
> categorical predictors, but its p-value was 0.0647, which is slightly
> above 0.05. It was retained because it had the highest mutual
> information among the categorical features, according to the final
> selection rule."

------------------------------------------------------------------------

# 21. Chi-Square Manual Demonstration

The notebook gives a complete manual Chi-square demonstration for:

**Dietary_Habits**

Observed categories:

-   Keto
-   Regular
-   Vegan
-   Vegetarian

Target categories:

-   Balanced Diet
-   High-Protein Diet
-   Low-Carb Diet
-   Low-Fat Diet

Manual result:

**Chi-square = 16.1043899**

**Degrees of freedom = 9**

Library verification:

**Chi-square = 16.1043899**

**p-value = 0.0647334**

**df = 9**

### Why is this good?

The manual statistic matches the library verification statistic.

This shows that the from-scratch calculation is consistent.

### Output

> "I implemented the Chi-square calculation manually and then verified
> the result using the statistical library. The statistic matched
> exactly, which validates my implementation."

------------------------------------------------------------------------

# 22. ANOVA Feature Selection

ANOVA is used for:

**Numerical feature vs categorical target**

The target has four meal-plan groups.

### Formula idea

`F = Variance Between Groups / Variance Within Groups`

A higher F-statistic means the between-group variation is larger
relative to the within-group variation.

### Key results

  Feature                      F-statistic    p-value Decision
  -------------------------- ------------- ---------- ----------
  BMI                             3.574740   0.013395 Keep
  Protein_Intake                  3.055339   0.027279 Keep
  Weight_kg                       2.639415   0.047879 Keep
  Height_cm                       1.456652   0.224360 Remove
  Blood_Sugar_Level               1.167594   0.320543 Remove
  Blood_Pressure_Systolic         0.884187   0.448444 Remove
  Age                             0.856105   0.463160 Remove
  Fat_Intake                      0.700180   0.551862 Remove
  Exercise_Frequency              0.637552   0.590778 Remove
  Sleep_Hours                     0.603006   0.613014 Remove
  Caloric_Intake                  0.457203   0.712218 Remove
  Carbohydrate_Intake             0.414644   0.742497 Remove
  Cholesterol_Level               0.249648   0.861630 Remove
  Blood_Pressure_Diastolic        0.192642   0.901459 Remove
  Daily_Steps                     0.117130   0.950085 Remove

### Selection rule

The notebook uses:

**p-value \< 0.05 → evidence for selection**

Therefore the numerical features retained by ANOVA are:

1.  `BMI`
2.  `Protein_Intake`
3.  `Weight_kg`

### Output

> "For numerical predictors, I used ANOVA because the target is
> categorical with multiple classes. BMI, Protein_Intake and Weight_kg
> had p-values below 0.05, so they were selected."

------------------------------------------------------------------------

# 23. ANOVA Demonstration -- BMI

The notebook performs a detailed manual ANOVA demonstration for BMI.

### Group means

  Target Class          Group Mean   Group Size
  ------------------- ------------ ------------
  Balanced Diet            28.8350         1005
  High-Protein Diet        27.8381          995
  Low-Carb Diet            28.0169          931
  Low-Fat Diet             28.7071         1069

Overall mean:

**28.362422**

Between-group sum of squares:

**736.194952**

Within-group sum of squares:

**274316.920874**

Degrees of freedom:

-   Between = 3
-   Within = 3996

F-statistic:

**3.57474**

p-value:

**0.013395**

### Interpretation

Because p \< 0.05, the notebook treats BMI as showing statistically
meaningful differences across the target groups under the stated ANOVA
selection rule.

### Output

> "For BMI, the ANOVA F-statistic is 3.57474 and the p-value is
> 0.013395. Since the p-value is below 0.05, BMI is retained."

------------------------------------------------------------------------

# 24. Mutual Information

Mutual Information measures how much information one variable provides
about another.

For the manual implementation, the notebook uses categorical/discrete
data.

### Important property

`MI = 0` indicates independence under the estimated distribution.

Higher MI means more shared information.

The notebook measures MI in:

**bits**

### Results

  Feature                 Mutual Information (bits)
  --------------------- ---------------------------
  Dietary_Habits                           0.002917
  Food_Aversions                           0.002233
  Allergies                                0.001848
  Preferred_Cuisine                        0.001653
  Chronic_Disease                          0.001379
  Genetic_Risk_Factor                      0.001006
  Gender                                   0.000765
  Smoking_Habit                            0.000437
  Alcohol_Consumption                      0.000123

### Most important result

The strongest categorical MI feature is:

**Dietary_Habits = 0.00291661 bits**

### Output

> "Mutual Information was used as an additional dependence measure for
> categorical features. Dietary_Habits had the highest MI at about
> 0.002917 bits, so it was retained by the final selection rule."

------------------------------------------------------------------------

# 25. Final Feature Selection

The final selected original features are:

1.  **Weight_kg**
2.  **BMI**
3.  **Protein_Intake**
4.  **Dietary_Habits**

### Why these four?

#### Weight_kg

Selected because:

**ANOVA p = 0.0478787 \< 0.05**

#### BMI

Selected because:

**ANOVA p = 0.0133946 \< 0.05**

#### Protein_Intake

Selected because:

**ANOVA p = 0.0272788 \< 0.05**

#### Dietary_Habits

Its Chi-square p-value was:

**0.0647334**

which is above 0.05.

However, it had the highest categorical MI:

**0.00291661 bits**

Therefore the final rule retained it.

### Output

> "After combining the statistical screening rules, my final four
> original features are Weight_kg, BMI, Protein_Intake and
> Dietary_Habits."

------------------------------------------------------------------------

# 26. Before vs After Feature Selection

The notebook reports:

  -----------------------------------------------------------------------
  Stage                                                Number of Features
  ------------------------------ ----------------------------------------
  Original Dataset                                                     30

  Original Predictors, target                                          29
  excluded                       

  Predictive Candidates after                                          24
  excluding ID and potential     
  target-derived outputs         

  Selected Original Features                                            4

  Removed Predictive Candidates                                        20
  -----------------------------------------------------------------------

### Why reduce 24 candidates to 4?

The goal of feature selection is to remove variables that do not provide
sufficient evidence under the selected statistical rules.

This can make the model:

-   simpler;
-   easier to interpret;
-   less redundant;
-   computationally lighter;
-   less exposed to irrelevant predictors.

### Output

> "I reduced the predictive candidate set from 24 features to four
> statistically supported original features. This gives a much simpler
> feature space."

------------------------------------------------------------------------

# 27. Leakage Prevention

The notebook deliberately excludes:

-   `Patient_ID`
-   `Recommended_Calories`
-   `Recommended_Protein`
-   `Recommended_Carbs`
-   `Recommended_Fats`

### Why?

The recommendation columns are outputs derived from the recommendation
process and could leak target-related information into a predictive
model.

`Patient_ID` is an identifier, not a meaningful predictive measurement.

### Output

> "I removed the identifier and recommendation-derived columns before
> feature selection and modeling to reduce the risk of data leakage."

------------------------------------------------------------------------

# 28. Train-Test Split

The notebook uses an **80/20 split** with random state 42.

  Dataset Part            Rows   Percentage
  -------------------- ------- ------------
  Full Modeling Data     5,000         100%
  Training               4,000          80%
  Testing                1,000          20%

### Why split?

The training data is used to learn preprocessing/feature information.

The testing data is held out for evaluating generalization.

### Output

> "I split the 5,000 records into 4,000 training observations and 1,000
> testing observations, using an 80/20 split with random state 42."

------------------------------------------------------------------------

# 29. Leakage-Safe Preprocessing Output

Before final feature selection, the transformed all-feature matrices
are:

-   Training matrix: **4000 × 45**
-   Testing matrix: **1000 × 45**

Missing values after preprocessing:

-   Training NaN = **0**
-   Testing NaN = **0**

After applying the selected feature set:

-   Selected training matrix: **4000 × 7**
-   Selected testing matrix: **1000 × 7**
-   Same encoded columns: **True**

### Why 7 columns from only 4 original features?

Because:

-   `Weight_kg` → 1 numerical column
-   `BMI` → 1 numerical column
-   `Protein_Intake` → 1 numerical column
-   `Dietary_Habits` → multiple one-hot categorical columns

Therefore, four original features become seven numerical/encoded model
columns.

### Output

> "The final four original features become seven model columns after
> one-hot encoding of Dietary_Habits. Both training and testing have
> exactly the same seven encoded columns."

------------------------------------------------------------------------

# 30. Before vs After Preprocessing

The notebook gives this final summary:

  Parameter                                 Before                            After
  ----------------------------- ------------------ --------------------------------
  Records                                     5000                             5000
  Features / columns                            30                                7
  Missing Values                              6765                                0
  Duplicate Records                              0                                0
  Categorical Features                          11                                0
  Potential IQR Outlier Flags                    4   Retained unless proven invalid
  Selected Original Features      Not yet selected                                4

### Important interpretation

The number of records remains 5,000.

The number of final model columns becomes 7.

Missing values become 0 after preprocessing.

Categorical features become numerical through encoding.

The four potential BMI IQR outliers are not automatically deleted.

### Output

> "The preprocessing pipeline keeps all 5,000 observations, removes
> missing values through training-based preprocessing, converts
> categorical variables to numerical representation, and reduces the
> final model matrix to seven columns based on four selected original
> features."

------------------------------------------------------------------------

# 31. Accuracy -- Very Important

## Does the uploaded notebook contain model accuracy?

**No.**

The uploaded `main_analysis.ipynb` contains:

-   data exploration;
-   missing-value analysis;
-   duplicate analysis;
-   encoding;
-   outlier analysis;
-   transformation;
-   scaling;
-   visualization;
-   train-test splitting;
-   leakage-safe preprocessing;
-   variance;
-   Pearson correlation;
-   Chi-square;
-   ANOVA;
-   Mutual Information;
-   final feature selection.

However, the uploaded notebook **does not contain a
classifier-training/evaluation section with an accuracy value**.

There is no completed accuracy output, confusion matrix, classification
report, or F1-score output in the uploaded notebook.

### Therefore, do not say:

> "The notebook achieved 24.9% accuracy."

unless that number comes from a separate model-training notebook/run.

### Correct Output

> "This notebook focuses on preprocessing and feature selection. It
> prepares the final training and testing matrices, but model training
> and accuracy evaluation are not included in the uploaded notebook."

------------------------------------------------------------------------

# 32. If Teacher Asks "Why Is There No Accuracy?"

Say:

> "The objective of this notebook is preprocessing and feature
> selection. The final selected dataset and leakage-safe train/test
> matrices are prepared here. Classification accuracy would be measured
> in the next modeling stage after training a classifier."

This is a better answer than inventing an accuracy value.

------------------------------------------------------------------------

# 33.  Graph Guide 

  -----------------------------------------------------------------------
  Graph             X-axis            Y-axis            Main Purpose
  ----------------- ----------------- ----------------- -----------------
  BMI Histogram     BMI intervals     Frequency         Distribution and
                                                        skewness

  Blood Sugar Box   Blood Sugar Level Value             Median,
  Plot                                                  quartiles,
                                                        extremes

  Meal Plan Bar     Meal-plan classes Number of         Target
  Chart                               observations      distribution

  Weight vs BMI     Weight (kg)       BMI               Relationship
  Scatter                                               between two
                                                        numerical
                                                        variables

  Pearson Heatmap   Numerical         Numerical         Correlation and
                    features          features          redundancy
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 34.  Statistical-Test Guide

  -------------------------------------------------------------------------
  Technique               Used For                Main Purpose
  ----------------------- ----------------------- -------------------------
  Variance                Numerical feature       Detect zero/very-low
                                                  variation

  Pearson Correlation     Numerical vs numerical  Detect linear
                                                  relationship/redundancy

  Chi-Square              Categorical vs          Test statistical
                          categorical             dependence

  ANOVA F-Test            Numerical feature vs    Compare numerical
                          categorical target      distributions across
                                                  target groups

  Mutual Information      Discrete/categorical    Measure shared
                          variables in manual     information/dependence
                          implementation          
  -------------------------------------------------------------------------


------------------------------------------------------------------------



# 37. Final Conclusion

The uploaded notebook successfully demonstrates a complete
**preprocessing and statistical feature-selection workflow**.

The strongest final points are:

1.  The raw dataset has **5,000 records and 30 columns**.
2.  Missing data was explicitly identified and treated.
3.  There are **0 exact duplicate records**.
4.  Potential outliers were detected but not blindly deleted.
5.  Categorical variables were encoded appropriately.
6.  Numerical variables were transformed/scaled where demonstrated.
7.  Multiple statistical feature-selection techniques were implemented.
8.  Leakage-risk columns were excluded.
9.  The feature space was reduced from **24 predictive candidates to 4
    selected original features**.
10. The final selected features are:

-   **Weight_kg**
-   **BMI**
-   **Protein_Intake**
-   **Dietary_Habits**

11. The final encoded matrices are:

-   **4000 × 7 training**
-   **1000 × 7 testing**

12. **Accuracy is not present in this uploaded
    preprocessing/feature-selection notebook.**

The most important conceptual lesson is:

> **Preprocessing makes the data suitable for machine learning, while
> feature selection chooses the most useful input variables. Model
> accuracy is a separate evaluation stage.**
