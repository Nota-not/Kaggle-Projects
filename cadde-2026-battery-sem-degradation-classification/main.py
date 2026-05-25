#---------------------------------------------------------------------------
# IMPORT LIBRARIES
#---------------------------------------------------------------------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

#---------------------------------------------------------------------------
# LOAD DATA
#---------------------------------------------------------------------------
train = pd.read_csv("/content/drive/MyDrive/Task_1_Datasets/train.csv")
test = pd.read_csv("/content/drive/MyDrive/Task_1_Datasets/test.csv")
targets = pd.read_csv("/content/drive/MyDrive/Task_1_Datasets/train_targets.csv")

#---------------------------------------------------------------------------
# Check data
#---------------------------------------------------------------------------

print(train.head())
print("-----------------------------------------------------------------")
print(test.head())
print("-----------------------------------------------------------------")
print(targets.head())

#---------------------------------------------------------------------------
# PREPARE (X) and TARGET (Y) (for regression)
#---------------------------------------------------------------------------

train = train.merge(targets, on="id", how="left")

target_col = [c for c in targets.columns if c!= "id"][0]
print(f"Predicting: '{target_col}'")
print(f"Target range: {train[target_col].min():.4f} to {train[target_col].max():.4f}")

#---------------------------------------------------------------------------
# UNDERSTANDING FEATURE AND TARGET RELATIONSHIPS
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Correlation Heatmap — see all feature-target relationships at once
#---------------------------------------------------------------------------
# (shows you which features are strongly correlated with your target)
# (if target correlation shows low score, can remove (redundancy))

# Correlation matrix
corr = train.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------
# Test if the target is skewed
#---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(train[target_col], kde=True)
plt.title(f"Distribution of Target: {target_col}")
plt.show()

#---------------------------------------------------------------------------
# Feature vs Target Scatterplot
#---------------------------------------------------------------------------
raw_features = ["Current(A)", "Voltage(V)", "Internal_Resistance(Ohm)", "dV/dt(V/s)"]

fig, axes = plt.subplots(1, len(raw_features), figsize=(18, 4))
for ax, col in zip(axes, raw_features):
    ax.scatter(train[col], train[target_col], alpha=0.3, s=10)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)
    ax.set_title(f"{col} vs Target")
plt.tight_layout()
plt.show()

