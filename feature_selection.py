"""From-scratch feature-selection routines used by the assignment."""
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f as f_dist

def manual_variance(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return 0.0 if len(s) == 0 else float(np.mean((s - s.mean()) ** 2))

def manual_pearson(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(float), y[mask].to_numpy(float)
    den = math.sqrt(np.sum((x-x.mean())**2) * np.sum((y-y.mean())**2))
    return 0.0 if den == 0 else float(np.sum((x-x.mean())*(y-y.mean()))/den)

def manual_chi_square(feature, target):
    x = pd.Series(feature).fillna("Unknown/Not Reported").astype(str)
    y = pd.Series(target).astype(str)
    observed = pd.crosstab(x, y)
    total = observed.values.sum()
    rows = observed.sum(axis=1).to_numpy()
    cols = observed.sum(axis=0).to_numpy()
    expected = np.zeros(observed.shape, float)
    for i in range(observed.shape[0]):
        for j in range(observed.shape[1]):
            expected[i,j] = rows[i] * cols[j] / total
    statistic = float(((observed.values-expected)**2 / np.where(expected==0,1,expected)).sum())
    dfree = int((observed.shape[0]-1)*(observed.shape[1]-1))
    p = float(chi2_contingency(observed)[1])
    return statistic, dfree, p

def manual_anova(groups):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    all_values = np.concatenate(groups)
    overall = all_values.mean()
    k, n = len(groups), len(all_values)
    ss_between = sum(len(g)*(g.mean()-overall)**2 for g in groups)
    ss_within = sum(np.sum((g-g.mean())**2) for g in groups)
    dfb, dfw = k-1, n-k
    msb, msw = ss_between/dfb, ss_within/dfw
    f_stat = float(msb/msw) if msw else float("inf")
    p = float(f_dist.sf(f_stat, dfb, dfw))
    return overall, ss_between, ss_within, dfb, dfw, f_stat, p

def manual_entropy(series):
    s = pd.Series(series).fillna("Unknown/Not Reported").astype(str)
    probs = s.value_counts(normalize=True)
    return float(-sum(p*math.log2(p) for p in probs if p > 0))

def manual_mutual_information(x, y):
    x = pd.Series(x).fillna("Unknown/Not Reported").astype(str)
    y = pd.Series(y).astype(str)
    joint = pd.crosstab(x, y, normalize=True)
    px, py = x.value_counts(normalize=True), y.value_counts(normalize=True)
    mi = 0.0
    for xv in joint.index:
        for yv in joint.columns:
            pxy = joint.loc[xv,yv]
            if pxy > 0:
                mi += pxy*math.log2(pxy/(px[xv]*py[yv]))
    return float(mi)

def select_features(train_df, target="Recommended_Meal_Plan"):
    """Replicates the notebook's final rule: ANOVA p<.05 for numeric,
    Chi-square p<.05 for categorical, plus strongest categorical MI;
    high-correlation numeric redundancy uses the stronger ANOVA feature."""
    exclude={target,"Patient_ID","Recommended_Calories","Recommended_Protein",
             "Recommended_Carbs","Recommended_Fats"}
    candidates=[c for c in train_df.columns if c not in exclude]
    numeric=train_df[candidates].select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in candidates if c not in numeric]
    classes=sorted(train_df[target].dropna().astype(str).unique())
    anova={}
    for c in numeric:
        groups=[train_df.loc[train_df[target].astype(str)==cl,c].dropna().to_numpy() for cl in classes]
        anova[c]=manual_anova(groups)[-1]
    corr=train_df[numeric].corr()
    redundant=set()
    for i,a in enumerate(numeric):
        for b in numeric[i+1:]:
            if abs(corr.loc[a,b]) >= .85:
                redundant.add(b if anova[a] <= anova[b] else a)
    selected_numeric=[c for c in numeric if anova[c] < .05 and c not in redundant]
    chi={c:manual_chi_square(train_df[c],train_df[target])[2] for c in categorical}
    selected_cat=[c for c in categorical if chi[c] < .05]
    mi={c:manual_mutual_information(train_df[c],train_df[target]) for c in categorical}
    if categorical:
        strongest=max(mi,key=mi.get)
        if strongest not in selected_cat:
            selected_cat.append(strongest)
    return selected_numeric+selected_cat, anova, chi, mi, redundant
