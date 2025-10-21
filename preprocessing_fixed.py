"""
Preprocessing Script for Tree Species Prediction in Sweden

This script loads four datasets (precipitation, soil, trees per area, crops),
merges them properly on 'county' and 'year', handles missing values intelligently,
encodes categorical features, performs feature selection, and saves the cleaned dataset.

Datasets:
- precipitation.xlsx: Yearly precipitation data
- soil.xlsx: Soil properties
- tress per area.xlsx: Trees per area (productive forest area)
- crops.xlsx: Growing stock and tree species

Output: final_dataset.csv in the AML-tree directory
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import os

# Set paths
data_dir = '/Users/godishalarishi/Desktop/Group 12'
output_dir = '/Users/godishalarishi/AML-tree'

# Load datasets (skip initial metadata rows)
print("Loading datasets...")
df_precipitation = pd.read_excel(os.path.join(data_dir, 'precipitation.xlsx'), header=None, skiprows=5)  # Adjust skiprows as needed
df_soil = pd.read_excel(os.path.join(data_dir, 'soil.xlsx'), header=None, skiprows=10)  # Adjust
df_forest_area = pd.read_excel(os.path.join(data_dir, 'tress per area.xlsx'), header=3)  # Data starts at row 3
df_crops = pd.read_excel(os.path.join(data_dir, 'crops.xlsx'), header=3)  # Data starts at row 3

# Rename columns based on inspection
df_forest_area.columns = ['year', 'county', 'table_contents', 'pine_forest', 'spruce_forest', 'lodgepole_pine_forest', 'mixed_conifer_forest', 'mixed_forest', 'broadleaf_forest', 'valuable_broadleaf_forest', 'bare_forest_land', 'all_forest_types']
df_crops.columns = ['year', 'region', 'diameter', 'tree_species', 'table_contents', 'incl_formally_protected', 'unnamed6', 'unnamed7', 'unnamed8', 'all_age_classes']

# For precipitation and soil, if they have data, rename accordingly (assuming they have year, county, precipitation)
if df_precipitation.shape[1] >= 3:
    df_precipitation.columns = ['year', 'county', 'precipitation']
if df_soil.shape[1] >= 2:
    df_soil.columns = ['year', 'county']  # Assuming soil has year and county, add soil columns if available

print("Datasets loaded. Columns:")
print(f"Precipitation: {df_precipitation.columns.tolist()}")
print(f"Soil: {df_soil.columns.tolist()}")
print(f"Forest Area: {df_forest_area.columns.tolist()}")
print(f"Crops: {df_crops.columns.tolist()}")

print("\nFirst few rows of each dataset:")
print("Precipitation head:")
print(df_precipitation.head())
print("\nSoil head:")
print(df_soil.head())
print("\nForest Area head:")
print(df_forest_area.head())
print("\nCrops head:")
print(df_crops.head())

# Standardize column names (strip spaces, lowercase)
for df in [df_precipitation, df_soil, df_forest_area, df_crops]:
    if hasattr(df.columns, 'str'):
        df.columns = df.columns.str.strip().str.lower()
    else:
        df.columns = [str(col).strip().lower() for col in df.columns]

# Rename columns for consistency
df_crops.rename(columns={'region': 'county'}, inplace=True)  # Rename region to county in crops

# Filter out empty datasets (precipitation and soil seem to have no data rows)
print("Precipitation dataset is empty or metadata, skipping.")
df_precipitation = pd.DataFrame()
print("Soil dataset is empty or metadata, skipping.")
df_soil = pd.DataFrame()

# Merge datasets on 'county' and 'year'
print("\nMerging datasets...")
merged_df = df_crops.merge(df_forest_area, on=['county', 'year'], how='outer', suffixes=('', '_forest'))
if not df_soil.empty:
    merged_df = merged_df.merge(df_soil, on=['county', 'year'], how='outer', suffixes=('', '_soil'))
if not df_precipitation.empty:
    merged_df = merged_df.merge(df_precipitation, on=['county', 'year'], how='outer', suffixes=('', '_precip'))

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Columns after merge: {merged_df.columns.tolist()}")

# Identify target column
target_col = 'tree_species'
if target_col not in merged_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in merged dataset.")

# Separate features and target
features = merged_df.drop(columns=[target_col])
target = merged_df[target_col]

# Identify categorical and numerical columns
categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Handle missing values
print("\nHandling missing values...")

# For numerical: KNN imputer
if numerical_cols:
    knn_imputer = KNNImputer(n_neighbors=5)
    features[numerical_cols] = knn_imputer.fit_transform(features[numerical_cols])

# For categorical: mode imputation
for col in categorical_cols:
    mode_val = features[col].mode()
    if not mode_val.empty:
        features[col].fillna(mode_val[0], inplace=True)

# For target: mode imputation
target.fillna(target.mode()[0], inplace=True)

print("Missing values handled.")

# Encode categorical features
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))
    label_encoders[col] = le

print("Categorical features encoded.")

# Feature selection
print("\nPerforming feature selection...")

# 1. Variance threshold (remove low variance features)
if numerical_cols:
    selector_var = VarianceThreshold(threshold=0.01)
    features_num = features[numerical_cols]
    features_num_selected = selector_var.fit_transform(features_num)
    selected_num_cols = [numerical_cols[i] for i in selector_var.get_support(indices=True)]
    features = features[selected_num_cols + categorical_cols]
    print(f"After variance threshold: {features.shape[1]} features")

# 2. Correlation threshold (remove highly correlated features)
if len(features.columns) > 1:
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    features.drop(columns=to_drop, inplace=True)
    print(f"After correlation threshold: {features.shape[1]} features, dropped: {to_drop}")

# 3. SelectKBest for classification (if target is available)
if len(features.columns) > 10:  # Select top 10 if many features
    selector_kbest = SelectKBest(score_func=f_classif, k=10)
    features_selected = selector_kbest.fit_transform(features, target)
    selected_cols = [features.columns[i] for i in selector_kbest.get_support(indices=True)]
    features = features[selected_cols]
    print(f"After SelectKBest: {features.shape[1]} features")

print("Feature selection complete.")

# Normalize numerical features (exclude 'county' as it's a categorical identifier)
print("\nNormalizing numerical features...")
scaler = MinMaxScaler()
num_cols_remaining = [col for col in features.columns if col in numerical_cols and col != 'county']
if num_cols_remaining:
    features[num_cols_remaining] = scaler.fit_transform(features[num_cols_remaining])

print("Normalization complete.")

# Recombine features and target
final_df = pd.concat([features, target], axis=1)

# Save final dataset
output_path = os.path.join(output_dir, 'final_dataset.csv')
final_df.to_csv(output_path, index=False)
print(f"\nFinal dataset saved to {output_path}")
print(f"Final shape: {final_df.shape}")
print("Preprocessing complete!")
