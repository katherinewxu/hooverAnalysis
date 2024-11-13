import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Load dataset with low_memory=False to handle mixed types
file_path = '~/Downloads/final_merged_data.csv'
data = pd.read_csv(file_path, low_memory=False)

# Remove columns with no observed values
data = data.dropna(axis=1, how="all")

# Define dependent variables and filter control columns for numeric data only
dependent_vars = ["working_now", "find_job"]
control_columns = [col for col in data.columns if col not in dependent_vars and col not in ["area", "sub_area"]]
numeric_control_columns = data[control_columns].select_dtypes(include=[float, int]).columns.tolist()

# Convert categorical variables `area` and `sub_area` to dummy variables for fixed effects and drop first column to prevent multicollinearity
if "area" in data.columns:
    data = pd.get_dummies(data, columns=["area"], prefix="area", drop_first=True)
if "sub_area" in data.columns:
    data = pd.get_dummies(data, columns=["sub_area"], prefix="sub_area", drop_first=True)

def select_features(X, y):
    # Keep only numeric columns in X
    X_numeric = X.select_dtypes(include=[float, int])
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.3, random_state=0)
    
    # Standardize features and handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Feature selection with Lasso
    lasso = LassoCV(cv=5, random_state=0).fit(X_train_scaled, y_train)
    
    # Align the coefficients with the correct columns
    lasso_coef = pd.Series(lasso.coef_, index=X_numeric.columns[:len(lasso.coef_)])
    selected_features_lasso = lasso_coef[lasso_coef != 0].index.tolist()
    
    # Feature selection with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    feature_importances = pd.Series(rf.feature_importances_, index=X_numeric.columns)
    selected_features_rf = feature_importances.nlargest(10).index.tolist()  # Select top 10 features by importance
    
    # Recursive Feature Elimination (RFE) with Random Forest
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=0), n_features_to_select=10)
    rfe.fit(X_train, y_train)
    selected_features_rfe = X_numeric.columns[rfe.support_].tolist()
    
    # Combine selected features
    combined_features = set(selected_features_lasso + selected_features_rf + selected_features_rfe)
    
    return list(combined_features), lasso_coef, feature_importances


# Prepare function for double-lasso feature selection
def double_lasso_feature_selection(X, y):
    # Keep only numeric columns in X
    X_numeric = X.select_dtypes(include=[float, int])
    
    # Preprocess pipeline with imputation and scaling
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_preprocessed = pipeline.fit_transform(X_numeric)
    
    # Lasso for feature selection
    lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_preprocessed, y)
    
    # Ensure the size of columns matches the number of coefficients
    if len(lasso.coef_) == X_numeric.shape[1]:
        selected_features = X_numeric.columns[lasso.coef_ != 0]
    else:
        # Align column names with the non-zero coefficients
        selected_features = X_numeric.columns[np.arange(len(lasso.coef_))[lasso.coef_ != 0]]
    
    return selected_features

# Find best predictors for each dependent variable
results = {}
for target_var in dependent_vars:
    print(f"\nSelecting features for target variable: {target_var}")
    
    # Drop rows with NaN values in target variable and separate target and predictors
    target_data = data.dropna(subset=[target_var])
    X = target_data.drop(columns=dependent_vars)  # Dropping all dependent variables, keeping only predictors
    y = target_data[target_var]
    
    # Feature selection
    selected_features, lasso_coef, rf_importances = select_features(X, y)
    
    # Show top predictors
    print(f"Selected features for {target_var}: {selected_features}")
    
    # Store results
    results[target_var] = {
        "selected_features": selected_features,
        "lasso_coef": lasso_coef,
        "rf_importances": rf_importances
    }

# Display regression results for each dependent variable
for var, res in results.items():
    print(f"\n=== Regression Results for {var} ===")
    print(res)
