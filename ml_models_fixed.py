"""
Machine Learning Models for Tree Species Prediction in Sweden

This script loads the preprocessed dataset, performs clustering with automatic k determination,
trains classification models incorporating cluster labels, evaluates them, and provides predictions.

Steps:
- Load final_dataset.csv
- Clustering: Determine optimal k using silhouette score and elbow method
- Classification: Train RandomForest and XGBoost with cluster as additional feature
- Evaluation: Accuracy, F1-score, confusion matrix, top-N accuracy
- Prediction function for best tree species per region/county

Output: Models trained and evaluated, prediction function available
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os

# Load preprocessed dataset
output_dir = '/Users/godishalarishi/AML-tree'
data_path = os.path.join(output_dir, 'final_dataset.csv')
df = pd.read_csv(data_path)

print("Loaded dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Unique counties in dataset:", df['county'].unique()[:10])  # Show first 10 unique counties

# County mapping (from original notebook)
county_mapping = {
    26: 'County A',
    14: 'Stockholms län',
    17: 'Uppsala län',
    16: 'Södermanlands län',
    25: 'Östergötlands län',
    7: 'Jönköpings län',
    9: 'Kronobergs län',
    8: 'Kalmar län',
    2: 'Gotlands län',
    0: 'Blekinge län',
    13: 'Skåne län',
    5: 'Hallands län',
    22: 'Västra Götalands län',
    18: 'Värmlands län',
    24: 'Örebro län',
    21: 'Västmanlands län',
    1: 'Dalarnas län',
    3: 'Gävleborgs län',
    20: 'Västernorrlands län',
    6: 'Jämtlands län',
    19: 'Västerbottens län',
    11: 'Norrbottens län',
    10: 'N Norrland',
    12: 'S Norrland',
    15: 'Svealand',
    4: 'Götaland',
    23: 'Whole country'
}

# Identify target and features
target_col = 'tree_species'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found.")

features = df.drop(columns=[target_col])
target = df[target_col]

# Encode target if not numeric
if target.dtype == 'object':
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(target)

# Clustering: Determine optimal number of clusters
print("\nDetermining optimal number of clusters...")
cluster_features = features.select_dtypes(include=[np.number]).columns.tolist()
if not cluster_features:
    cluster_features = features.columns[:2]  # Fallback

X_cluster = features[cluster_features]

# Elbow method and silhouette
inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_cluster, kmeans.labels_))

# Optimal k: max silhouette or elbow point (simple heuristic: where inertia drops less)
optimal_k = k_range[np.argmax(sil_scores)]
print(f"Optimal k based on silhouette: {optimal_k}")

# Perform clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)

# Evaluate clustering
sampled_df = resample(df, n_samples=min(5000, len(df)), random_state=42)
sil_score = silhouette_score(sampled_df[cluster_features], sampled_df['cluster'])
db_index = davies_bouldin_score(df[cluster_features], df['cluster'])
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")

# Add cluster as feature for classification
features['cluster'] = df['cluster']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train models
print("\nTraining models...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("\nEvaluation Metrics:")

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    top3_acc = top_k_accuracy_score(y_true, rf_model.predict_proba(X_test), k=3)  # Top-3 accuracy
    print(f"\n{model_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred, zero_division=1))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Tree species mapping (assuming standard mapping; adjust if needed)
tree_species_mapping = {
    0: 'Pine', 1: 'Spruce', 2: 'Lodgepole Pine',
    3: 'Mixed Conifer', 4: 'Broadleaf', 5: 'Valuable Broadleaf'
}

# Prediction function
def predict_best_tree_species(county_name, region_col='county'):
    """
    Predict the best tree species for a given county/region based on trained models.
    Returns top 3 recommended species.
    """
    if region_col not in df.columns:
        return f"Region column '{region_col}' not found."

    # Handle county name: if it's a string, find the corresponding numeric code
    if isinstance(county_name, str):
        # Reverse mapping to get numeric code from name
        reverse_mapping = {v: k for k, v in county_mapping.items()}
        county_code = reverse_mapping.get(county_name)
        if county_code is None:
            return f"County name '{county_name}' not found in mapping."
        county_value = county_code
    else:
        county_value = county_name

    # Filter data for the region
    region_data = df[df[region_col] == county_value]
    if region_data.empty:
        return f"No data available for '{county_name}' (code: {county_value})."

    # Get features for prediction (average or first row)
    region_features = region_data.drop(columns=[target_col, 'cluster']).mean().to_frame().T  # Average features
    # Add cluster: use the most common cluster or average
    region_features['cluster'] = region_data['cluster'].mode()[0] if not region_data['cluster'].mode().empty else region_data['cluster'].mean()

    # Predict probabilities
    rf_probs = rf_model.predict_proba(region_features)[0]
    xgb_probs = xgb_model.predict_proba(region_features)[0]

    # Average probabilities
    avg_probs = (rf_probs + xgb_probs) / 2

    # Get top 3 indices
    top_indices = np.argsort(avg_probs)[-3:][::-1]
    top_species = [tree_species_mapping.get(i, f'Species {i}') for i in top_indices]

    return f"Top tree species for {county_name}: {', '.join(top_species)}"

# Example predictions
print("\nExample Predictions:")
counties_to_test = ['Stockholms län', 'Uppsala län', 'Gotlands län']
for county in counties_to_test:
    result = predict_best_tree_species(county)
    print(result)

print("\nML models script complete!")
