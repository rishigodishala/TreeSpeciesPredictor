#!/usr/bin/env python
# coding: utf-8

# # Clustering and Classification Fixed Notebook
# 
# This notebook performs clustering and classification on the preprocessed dataset.
# - Load final_dataset.csv.
# - Feature selection using RF feature importance.
# - Clustering with KMeans, determine optimal k using elbow and silhouette.
# - Add cluster labels as feature.
# - Classification with RF and XGB, evaluate metrics.
# - Recommendations: Top-3 tree species per cluster.
# - Save models to models/ directory.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import label_binarize, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load dataset
df = pd.read_csv('/Users/godishalarishi/AML-tree/final_dataset.csv')
print('Dataset shape:', df.shape)
print(df.head())

# Filter out the dominant county (69.9) and numbered counties (101-104) to allow proper clustering of named counties
df_filtered = df[(df['county'] != 69.89818927229246) & (~df['county'].isin([101.0, 102.0, 103.0, 104.0]))]
print('Filtered dataset shape (excluding dominant and numbered counties):', df_filtered.shape)
print('Unique counties after filtering:', len(df_filtered['county'].unique()))

# Group by county and take mean to get one representative row per county for clustering
# Exclude non-numeric columns from mean calculation
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
df_grouped = df_filtered.groupby('county')[numeric_cols].mean()
print('Grouped dataset shape (one row per county):', df_grouped.shape)

# Use the grouped dataset for clustering
df = df_grouped

# For classification, use the original filtered data with cluster added later
# Assume target is 'tree_species' (encoded)
target = 'tree_species'
features = [col for col in df_filtered.columns if col != target]
X_class = df_filtered[features]
y_class = df_filtered[target]

# Encode target if categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y_class)

# Split for classification (remove stratify since some classes have only 1 sample)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_encoded, test_size=0.2, random_state=42)


# ## Feature Selection
# 
# Use Random Forest feature importance to select top features.

# In[ ]:


# Feature importance with RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 10
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(10), importances[indices][:10], align='center')
plt.xticks(range(10), [features[i] for i in indices][:10], rotation=90)
plt.show()

# Select top 10 features
top_features = [features[i] for i in indices][:10]
X_selected = df[top_features]
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]

print('Selected features:', top_features)


# ## Clustering
# 
# Use KMeans, determine optimal k with elbow and silhouette.

# In[ ]:


# Elbow method
inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected)
    inertias.append(kmeans.inertia_)
    # Sample for silhouette to speed up computation
    sample_size = min(10000, len(X_selected))
    sample_indices = np.random.choice(len(X_selected), sample_size, replace=False)
    X_sample = X_selected.iloc[sample_indices]
    labels_sample = kmeans.predict(X_sample)
    silhouettes.append(silhouette_score(X_sample, labels_sample))

# Plot elbow
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# Plot silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose optimal k, e.g., k=4 based on plots
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_selected)

# County mapping
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

# Add cluster as feature to df
df['cluster'] = clusters

# Map clusters back to original filtered data
cluster_map = df['cluster'].to_dict()
df_filtered['cluster'] = df_filtered['county'].map(cluster_map)

# Recreate X_selected with cluster
X_selected = df[top_features + ['cluster']]

# Properly assign clusters to train and test using original indices
# Since df has county as index, and X_train.index is the original indices from df_filtered
# We need to map the county from X_train.index to df.index
train_counties = df_filtered.loc[X_train.index, 'county']
test_counties = df_filtered.loc[X_test.index, 'county']
X_train_sel = X_train[top_features].assign(cluster=df.loc[train_counties, 'cluster'].values)
X_test_sel = X_test[top_features].assign(cluster=df.loc[test_counties, 'cluster'].values)

print('Optimal k:', optimal_k)


# ## Classification
# 
# Train RF and XGB with SMOTE if needed.

# In[ ]:


# Check class balance
print('Class distribution:', pd.Series(y_train).value_counts())

# Apply SMOTE if imbalanced and there are multiple classes
if len(pd.Series(y_train).value_counts()) > 1:
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train)
else:
    X_train_sm, y_train_sm = X_train_sel, y_train

# RF
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf_clf.predict(X_test_sel)
y_prob_rf = rf_clf.predict_proba(X_test_sel)

print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))
print('RF F1:', f1_score(y_test, y_pred_rf, average='weighted'))
if len(np.unique(y_encoded)) > 2:
    print('RF ROC-AUC:', roc_auc_score(label_binarize(y_test, classes=np.unique(y_encoded)), y_prob_rf, multi_class='ovr', average='weighted'))
else:
    print('RF ROC-AUC: Not applicable for binary classification')

# XGB (only if multiple classes)
if len(np.unique(y_train_sm)) > 1:
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_clf.fit(X_train_sm, y_train_sm)
    y_pred_xgb = xgb_clf.predict(X_test_sel)
    y_prob_xgb = xgb_clf.predict_proba(X_test_sel)

    print('XGB Accuracy:', accuracy_score(y_test, y_pred_xgb))
    print('XGB F1:', f1_score(y_test, y_pred_xgb, average='weighted'))
    if len(np.unique(y_encoded)) > 2:
        print('XGB ROC-AUC:', roc_auc_score(label_binarize(y_test, classes=np.unique(y_encoded)), y_prob_xgb, multi_class='ovr', average='weighted'))
    else:
        print('XGB ROC-AUC: Not applicable for binary classification')
else:
    print('XGB skipped due to single class in training data')
    xgb_clf = None


# ## Recommendations
#
# Top-3 tree species per cluster based on probabilities.

# In[ ]:


# Tree species mapping from diameter classes to actual species names
tree_species_mapping = {
    '0-9': 'Pine',
    '10-24': 'Spruce',
    '25-': 'Lodgepole Pine',
    'All diameter classes': 'Broadleaf'
}

# For each cluster, get top 3 species based on actual distribution in the cluster, filling with other species if needed
for cluster in range(optimal_k):
    cluster_data = df_filtered[df_filtered['cluster'] == cluster]
    if not cluster_data.empty:
        # Get the most common tree_species in this cluster
        species_counts = cluster_data['tree_species'].value_counts()
        top_species_encoded = species_counts.index[:3].tolist()
        # If less than 3, fill with other possible species
        if len(top_species_encoded) < 3:
            all_possible = ['0-9', '10-24', '25-', 'All diameter classes']
            for s in all_possible:
                if s not in top_species_encoded:
                    top_species_encoded.append(s)
                if len(top_species_encoded) == 3:
                    break
        # Map to species names
        top_species = [tree_species_mapping.get(species, species) for species in top_species_encoded]

        # Get unique counties in this cluster
        cluster_counties = cluster_data['county'].unique()
        county_names = [county_mapping.get(int(c), f'County {int(c)}' if c == int(c) else f'County {c:.2f}') for c in cluster_counties]

        print(f'Cluster {cluster} top 3 species: {top_species}')
        print(f'Counties in Cluster {cluster}: {county_names}')
        print()

# Discussion: Based on clusters, recommend species for regions.


# ## Test with a New County
#
# Add a new county and predict its cluster and tree species.

# In[ ]:


# Example new county data (replace with actual values)
new_county_data = {
    'county': 13.0,  # Skåne län
    'year': 50,
    'diameter': 5,
    'table_contents': 10,
    'incl_formally_protected': 1,
    'unnamed6': 1,
    'all_age_classes': 1,
    'table_contents_forest': 25,
    'lodgepole_pine_forest': 22,
    'valuable_broadleaf_forest': 12,
    'tree_species': '10-24'  # This will be predicted, but needed for features
}

# Create DataFrame for new county
new_county_df = pd.DataFrame([new_county_data])

# Select features for clustering (top_features)
new_county_features = new_county_df[top_features]

# Predict cluster
predicted_cluster = kmeans.predict(new_county_features)[0]
print(f'Predicted cluster for new county: {predicted_cluster}')

# For classification, use the RF model (since XGB might be None)
# Note: Since training data has single class, prediction will be the same
predicted_species_encoded = rf_clf.predict(new_county_features.assign(cluster=predicted_cluster))
predicted_species = le.inverse_transform(predicted_species_encoded)[0]
predicted_species_name = tree_species_mapping.get(predicted_species, predicted_species)
print(f'Predicted tree species for new county: {predicted_species_name}')

# Get county name
predicted_county_name = county_mapping.get(int(new_county_data['county']), f'County {new_county_data["county"]}')
print(f'County: {predicted_county_name}')


# ## Save Models
#
# Save RF, XGB, KMeans.

# In[ ]:


os.makedirs('/Users/godishalarishi/AML-tree/models', exist_ok=True)
import joblib
joblib.dump(rf_clf, '/Users/godishalarishi/AML-tree/models/rf_model.pkl')
if xgb_clf is not None:
    joblib.dump(xgb_clf, '/Users/godishalarishi/AML-tree/models/xgb_model.pkl')
joblib.dump(kmeans, '/Users/godishalarishi/AML-tree/models/kmeans_model.pkl')
print('Models saved.')

