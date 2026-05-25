#-----------------------------------------------------------------------------
# IMPORT LIBRARIES
#-----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------
# LOAD DATA
#---------------------------------------------------------------------------

train = pd.read_csv("/content/drive/MyDrive/Task_2_Datasets/train.csv")
test = pd.read_csv("/content/drive/MyDrive/Task_2_Datasets/test.csv")
labels = pd.read_csv("/content/drive/MyDrive/Task_2_Datasets/train_labels.csv")

# Insert id column for train and test data
train.insert(0, "id", range(len(train)))
test.insert(0, "id", range(len(test)))

#---------------------------------------------------------------------------
# Check data
#---------------------------------------------------------------------------
print(train.head())
print("-----------------------------------------------------------------")
print(test.head())
print("-----------------------------------------------------------------")
print(labels.head())

#---------------------------------------------------------------------------
# Correlation Heatmap - see all feature-target relationships at once
#---------------------------------------------------------------------------
# (shows you which features are strongly related with target)
# (if target shows low score, can remove)

raw_corr_df = X.copy()  # X still has only the 5 raw features at this point
raw_corr_df["Anomaly"] = y.values

feature_target_corr = raw_corr_df.corr(numeric_only=True)

plt.figure(figsize=(6, 5))
sns.heatmap(feature_target_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature + Target Correlation Matrix (Raw Features)")
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------
# ADD ENGINEERING FEATURES
#----------------------------------------------------------------------------
# Only important features identified from the feature hunt are added

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pressure differentials
    df["pressure_drop"]       = df["Inlet_Pressure"]  - df["Outlet_Pressure"]
    df["pump_inlet_diff"]     = df["Pump_Pressure"]   - df["Inlet_Pressure"]
    df["pump_outlet_diff"]    = df["Pump_Pressure"]   - df["Outlet_Pressure"]
    df["total_pressure_diff"] = df["Inlet_Pressure"]  - df["Outlet_Pressure"]

    # Efficiency-like ratios (guard against division by zero)
    eps = 1e-9
    df["pressure_ratio"]      = df["Outlet_Pressure"] / (df["Inlet_Pressure"] + eps)
    df["pump_efficiency"]     = df["Outlet_Pressure"] / (df["Pump_Pressure"]  + eps)
    df["flow_per_pressure"]   = df["Flowrate"]        / (df["Pump_Pressure"]  + eps)

    # Thermal interactions
    df["temp_x_flow"]         = df["Temperature"]    * df["Flowrate"]
    df["temp_x_pump_press"]   = df["Temperature"]    * df["Pump_Pressure"]
    df["temp_x_pressure_drop"]= df["Temperature"]    * df["pressure_drop"]

    # Polynomial / nonlinear
    df["pressure_drop_sq"]    = df["pressure_drop"]  ** 2
    df["flowrate_sq"]         = df["Flowrate"]       ** 2
    df["temp_sq"]             = df["Temperature"]    ** 2

    # Log transforms (shift to ensure positive)
    for col in ["Temperature", "Pump_Pressure", "Inlet_Pressure",
                "Outlet_Pressure", "Flowrate"]:
        shifted = df[col] - df[col].min() + 1
        df[f"log_{col}"] = np.log(shifted)

    # Rolling statistics (sorted by id if available, else just use index)
    for col in ["Flowrate", "Temperature", "pressure_drop"]:
        df[f"{col}_rolling_mean3"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_rolling_std3"]  = df[col].rolling(3, min_periods=1).std().fillna(0)

    return df

X      = engineer_features(X)
X_test = engineer_features(X_test)