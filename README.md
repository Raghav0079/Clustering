# Clustering Interface

## Overview
This project provides an interactive interface for clustering wine quality data using various unsupervised machine learning algorithms. The main goal is to help users explore clustering techniques and visualize their results on the wine dataset. The project is implemented in Python using Streamlit for the web interface and includes a Jupyter notebook for deeper analysis.

## Features
- **Upload CSV Dataset:** Users can upload their own CSV files (e.g., `winequality-red.csv`).
- **Data Preview:** View the first few rows of the uploaded dataset.
- **Clustering Algorithms:**
  - **KMeans:** Partition data into a specified number of clusters.
  - **DBSCAN:** Density-based clustering, automatically detects clusters and outliers.
  - **Gaussian Mixture Model (GMM):** Probabilistic clustering using mixture of Gaussians.
- **Parameter Selection:** Adjust algorithm parameters interactively (e.g., number of clusters, DBSCAN epsilon).
- **Visualization:** Scatter plot of clusters using the first two features.
- **Download Results:** Export the clustered data as a CSV file.

## How It Works
1. **Run the App:**
   - Install dependencies: `pip install streamlit pandas scikit-learn matplotlib`
   - Start the app: `streamlit run app.py`
2. **Upload Data:**
   - Use the file uploader to select your CSV file (default: `winequality-red.csv`).
3. **Select Algorithm:**
   - Choose between KMeans, DBSCAN, or GMM.
   - Adjust parameters using sliders.
4. **View Results:**
   - See cluster assignments and inertia/components.
   - Visualize clusters in a scatter plot.
   - Download the clustered data for further analysis.

## File Structure
- `app.py`: Streamlit web app for interactive clustering.
- `Wine Quality.ipynb`: Jupyter notebook with step-by-step clustering analysis and visualizations.
- `winequality-red.csv`: Sample wine quality dataset.
- `README.md`: Project documentation.

## Example Dataset
The wine dataset contains the following columns:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality

## Algorithms Explained
### KMeans
- Partitions data into K clusters by minimizing within-cluster variance.
- User selects number of clusters.
- Outputs inertia (sum of squared distances to cluster centers).

### DBSCAN
- Groups points that are closely packed together.
- Detects outliers as points in low-density regions.
- User sets `eps` (distance threshold) and `min_samples`.

### Gaussian Mixture Model (GMM)
- Models data as a mixture of several Gaussian distributions.
- User selects number of components.
- Outputs number of components used.

## Jupyter Notebook
The notebook (`Wine Quality.ipynb`) provides:
- Data exploration and visualization (histograms, bar plots).
- Step-by-step implementation of KMeans, DBSCAN, and GMM.
- Code examples for clustering and plotting results.

## Getting Started
1. Clone the repository.
2. Install required packages.
3. Run the Streamlit app or open the notebook for analysis.

## Requirements
- Python 3.7+
- streamlit
- pandas
- scikit-learn
- matplotlib

## License
This project is open source and free to use for educational purposes.

## Author
Created by Raghav0079
