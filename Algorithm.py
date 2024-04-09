"""
Name: 
project1:
file created : April 8 2024
Last modified : April 8 2024 - Sajjad
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from numpy.linalg import pinv

class DNAMissingValueEstimator:
    def __init__(self):
        self.data = None
        self.imputed_data = None

    def load_data(self, file_path):
        """
        Load DNA sequence data from a file.
        Assumes the data is in a tab-separated format with rows representing samples and columns representing features.
        """
        try:
            self.data = pd.read_csv(file_path, sep='\t')
            print(f"Loaded {len(self.data)} samples with {len(self.data.columns)} features.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def preprocess_data(self, window_size=None, k_neighbors=None, missing_percentage=None):
        """
        Preprocess the DNA data (e.g., handle missing values, normalize, etc.) using all three methods.
        """
        # Make a copy of the original data
        self.imputed_data = self.data.copy()

        # Simulate missing values
        self.simulate_missing_values(missing_percentage)

        # Perform imputation using different methods
        self.dynamic_local_least_imputation(window_size)
        self.local_least_imputation(window_size)
        self.knn_imputation(k_neighbors)

    def evaluate_performance(self, true_data):
        """
        Evaluate the performance of the imputation methods using RMSE and NRMSE.
        """
        for method in self.imputed_data.columns:
            imputed_values = self.imputed_data[method]
            rmse = np.sqrt(mean_squared_error(true_data, imputed_values))
            range_true = np.max(true_data) - np.min(true_data)
            nrmse = rmse / range_true
            print(f"Method: {method}")
            print(f"RMSE: {rmse:.4f}")
            print(f"NRMSE: {nrmse:.4f}")
            print()

    def simulate_missing_values(self, missing_percentage):
        """
        Simulate missing values in the data.
        """
        num_missing = int(len(self.data) * len(self.data.columns) * missing_percentage / 100)
        indices = np.random.choice(len(self.data), num_missing, replace=True)
        columns = np.random.choice(self.data.columns, num_missing, replace=True)
        self.imputed_data.loc[indices, columns] = np.nan

    def dynamic_local_least_imputation(self, window_size):
        """
        Implement dynamic local least imputation.
        """
        self.imputed_data['Dynamic Local Least Imputation'] = self.imputed_data.interpolate(method='polynomial', order=1, axis=1)

    def local_least_imputation(self, window_size):
        """
        Implement local least imputation.
        """
        self.imputed_data['Local Least Imputation'] = self.imputed_data.interpolate(method='linear', axis=1)

    def knn_imputation(self, k_neighbors):
        """
        Implement KNN imputation.
        """
        imputer = KNNImputer(n_neighbors=k_neighbors)
        self.imputed_data['KNN Imputation'] = pd.DataFrame(imputer.fit_transform(self.imputed_data), columns=self.imputed_data.columns)

    def moore_penrose_checks(self):
        """
        Perform Moore-Penrose inverse checks on the imputed data.
        """
        for method, imputed_values in self.imputed_data.items():
            pinv_result = pinv(imputed_values)
            print(f"Moore-Penrose inverse check for {method}:")
            print(pinv_result)

    # Add other methods or helper functions as needed

if __name__ == "__main__":
    # Instantiate your DNAMissingValueEstimator
    estimator = DNAMissingValueEstimator()

    # Load data from your file 
    data_file_path = "chimpanzee.txt"
    estimator.load_data(data_file_path)

    # Define parameters
    window_size = 5
    k_neighbors = 5
    missing_percentage = 10  # Example: 10% missing values

    # Preprocess data with different methods and parameters
    estimator.preprocess_data(window_size=window_size, k_neighbors=k_neighbors, missing_percentage=missing_percentage)

    # Load true data (replace with your actual true data)
    true_data = pd.read_csv("true_dna_data.csv", sep='\t')

    # Evaluate performance for each imputation method
    estimator.evaluate_performance(true_data)

    # Perform Moore-Penrose inverse checks
    estimator.moore_penrose_checks()

    # Add any additional steps or custom methods as required
