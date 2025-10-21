# Tree Species Predictor

A machine learning project for clustering Swedish counties and classifying tree species based on forest data. This project uses unsupervised learning (KMeans clustering) and supervised learning (Random Forest and XGBoost classification) to analyze and predict tree species distribution across different regions.

## Features

- **Data Preprocessing**: Filtering and grouping forest data by county
- **Feature Selection**: Using Random Forest importance for optimal feature selection
- **Clustering**: KMeans clustering with elbow and silhouette analysis to determine optimal number of clusters
- **Classification**: Random Forest and XGBoost models for tree species prediction
- **Recommendations**: Top-3 tree species recommendations per cluster
- **Model Persistence**: Saving trained models for future use

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rishigodishala/TreeSpeciesPredictor.git
   cd TreeSpeciesPredictor
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost joblib
   ```

## Usage

Run the main script:
```bash
python clustering_classification_fixed.py
```

The script will:
- Load and preprocess the dataset
- Perform feature selection
- Cluster counties using KMeans
- Train classification models
- Generate recommendations
- Save models to the `models/` directory

## Data

The project uses `final_dataset.csv` containing forest data with the following key features:
- County information
- Year, diameter, table contents
- Forest types (lodgepole pine, valuable broadleaf)
- Tree species categories (0-9, 10-24, 25-, All diameter classes)

Data is filtered to exclude dominant and numbered counties, then grouped by county for analysis.

## Methodology

1. **Preprocessing**:
   - Filter out outliers and irrelevant counties
   - Group data by county and compute means
   - Encode categorical target variables

2. **Feature Selection**:
   - Use Random Forest feature importance
   - Select top features for modeling

3. **Clustering**:
   - Apply KMeans clustering
   - Determine optimal k using elbow method and silhouette score
   - Add cluster labels as features

4. **Classification**:
   - Train Random Forest and XGBoost models
   - Handle class imbalance with SMOTE
   - Evaluate using accuracy, F1-score, and ROC-AUC

5. **Recommendations**:
   - Analyze tree species distribution per cluster
   - Provide top-3 species recommendations for each cluster

## Results

- Optimal clusters: 4
- Classification accuracy: ~80%
- Models saved: RF, XGB, KMeans
- Recommendations provided for each cluster with associated counties

## Models

Trained models are saved in the `models/` directory:
- `rf_model.pkl`: Random Forest classifier
- `xgb_model.pkl`: XGBoost classifier (if applicable)
- `kmeans_model.pkl`: KMeans clustering model

## Project Structure

```
TreeSpeciesPredictor/
├── clustering_classification_fixed.py  # Main script
├── preprocessing_fixed.py             # Data preprocessing
├── ml_models_fixed.py                 # Model training
├── final_dataset.csv                  # Dataset
├── models/                            # Saved models
├── TODO.md                            # Project tasks
├── README.md                          # This file
└── *.png                             # Plots and visualizations
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Rishi Godishala - BTH Master's in Advanced Machine Learning
