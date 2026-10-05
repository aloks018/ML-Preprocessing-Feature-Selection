"""Run the complete preprocessing + feature-selection workflow and save artifacts."""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from feature_selection import select_features

DATA=ROOT/"dataset"/"raw_dataset.csv"
OUT=ROOT/"results"/"outputs"; GRAPH=ROOT/"results"/"graphs"
OUT.mkdir(parents=True,exist_ok=True); GRAPH.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(DATA)
df=df.drop_duplicates().reset_index(drop=True)
rng=np.random.default_rng(42)
indices=np.arange(len(df)); rng.shuffle(indices)
test_count=int(round(len(df)*0.20))
test=df.iloc[indices[:test_count]].reset_index(drop=True)
train=df.iloc[indices[test_count:]].reset_index(drop=True)

selected, anova, chi, mi, redundant=select_features(train)
pd.DataFrame({"Feature":list(anova),"ANOVA_p":[anova[x] for x in anova]}).sort_values("ANOVA_p").to_csv(OUT/"anova_results.csv",index=False)
pd.DataFrame({"Feature":list(chi),"ChiSquare_p":[chi[x] for x in chi],"MI_bits":[mi[x] for x in chi]}).sort_values("ChiSquare_p").to_csv(OUT/"categorical_selection.csv",index=False)
pd.DataFrame({"Feature":selected,"Decision":["Keep"]*len(selected)}).to_csv(OUT/"selected_features.csv",index=False)

X=train[selected].copy(); y=train["Recommended_Meal_Plan"]
Xtest=test[selected].copy(); ytest=test["Recommended_Meal_Plan"]
num=X.select_dtypes(include=np.number).columns.tolist()
cat=[c for c in selected if c not in num]
pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
model=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=2000,random_state=42))])
model.fit(X,y); pred=model.predict(Xtest)
metrics={"accuracy":float(accuracy_score(ytest,pred)),
         "macro_f1":float(f1_score(ytest,pred,average="macro")),
         "weighted_f1":float(f1_score(ytest,pred,average="weighted"))}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
(OUT/"classification_report.txt").write_text(classification_report(ytest,pred))

# target distribution
counts=df["Recommended_Meal_Plan"].value_counts()
plt.figure(figsize=(8,5)); plt.bar(counts.index,counts.values); plt.xticks(rotation=20); plt.ylabel("Observations"); plt.title("Meal Plan Class Distribution"); plt.tight_layout(); plt.savefig(GRAPH/"class_distribution.png",dpi=160); plt.close()
# selection p-values
pvals=pd.DataFrame({"Feature":list(anova),"ANOVA p-value":[anova[x] for x in anova]}).sort_values("ANOVA p-value").head(15)
plt.figure(figsize=(9,6)); plt.barh(pvals["Feature"],pvals["ANOVA p-value"]); plt.gca().invert_yaxis(); plt.xlabel("p-value"); plt.title("ANOVA Feature-Target Evidence"); plt.tight_layout(); plt.savefig(GRAPH/"anova_pvalues.png",dpi=160); plt.close()

print("Selected:",selected); print("Metrics:",metrics)
