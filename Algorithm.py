"""
CSC571 Data Mining
Project 1
Programmers: Jeray Neely-Speaks, Sajjad Alsaffar, Abigail Garrido
Professor: Dr. Jeonghwa Lee
File Created: 4/8/2024
File Updated: 4/9/2024
"""

"""
note for you guys:
the idea is to copy the DNA make x percentage of the data missing and then impute the missing values using 3 different methods,
1. dynamic local least imputation
2. local least imputation
3. KNN imputation
and then evaluate the performance of the imputation methods using RMSE and NRMSE. ** need to get RMSE to do NRMSE.
so far it works fine but when it comes to evaluating the performance of the imputation methods using RMSE and NRMSE, I am still getting an error.
so I need to fix that.
some solution that i haven't tried are:
1. instead of using the .values.flatten() method, I can use the .values.ravel() method.
2. use slice of the chimpanzee data instead of the whole DNA sequence.
3. use the .iloc method to access values for imputation and assign back to the DataFrame.
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

        print("Before simulating missing values:", len(self.imputed_data))

        # Simulate missing values
        self.simulate_missing_values(missing_percentage)

        print("After simulating missing values:", len(self.imputed_data))

        # Convert DNA sequences to numeric format
        self.convert_to_numeric()

        print("After converting to numeric:", len(self.imputed_data))

        # Perform imputation using different methods
        self.dynamic_local_least_imputation(window_size)
        self.local_least_imputation(window_size)
        self.knn_imputation(k_neighbors)

        print("After imputation:", len(self.imputed_data))


    def convert_to_numeric(self):
        """
        Convert DNA sequences to numeric format using one-hot encoding.
        """
        # Define mapping of DNA bases to numeric values
        base_to_numeric = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        # Apply one-hot encoding to each DNA sequence column
        for col in self.imputed_data.columns:
            self.imputed_data[col] = self.imputed_data[col].apply(lambda x: [base_to_numeric.get(base, -1) for base in x])

        # Convert the DataFrame to numeric dtype
        self.imputed_data = self.imputed_data.apply(pd.to_numeric, errors='coerce')

    def simulate_missing_values(self, missing_percentage):
        """
        Simulate missing values in the data.
        """
        num_missing = int(len(self.data) * len(self.data.columns) * missing_percentage / 100)
        indices = np.random.choice(len(self.data), num_missing, replace=True)
        columns = np.random.choice(self.data.columns, num_missing, replace=True)
        self.imputed_data.loc[indices, columns] = np.nan

    def dynamic_local_least_imputation(self):
        """
        Implement dynamic local least imputation using linear interpolation.
        """
        # Iterate over each column and perform linear interpolation
        for col in self.imputed_data.columns:
            self.imputed_data[col] = self.imputed_data[col].interpolate(method='linear')

    def local_least_imputation(self):
        """
        Implement local least imputation using linear interpolation.
        """
        # Iterate over each column and perform linear interpolation
        for col in self.imputed_data.columns:
            self.imputed_data[col] = self.imputed_data[col].interpolate(method='linear')

    def knn_imputation(self, k_neighbors):
        """
        Implement KNN imputation.
        """
        imputer = KNNImputer(n_neighbors=k_neighbors)
        # Use `iloc` to access values for imputation and assign back to the DataFrame
        self.imputed_data.iloc[:, :] = imputer.fit_transform(self.imputed_data)

    def evaluate_performance(self, true_data):
        """
        Evaluate the performance of the imputation methods using RMSE and NRMSE.
        """
        # Get the number of rows in true data
        num_samples_true = true_data.shape[0]

        for method in self.imputed_data.columns:
            # Get the number of rows in imputed data for the current method
            num_samples_imputed = self.imputed_data[method].shape[0]

            # Ensure both datasets have the same number of samples
            if num_samples_true != num_samples_imputed:
                raise ValueError("Number of samples in true data and imputed data are not consistent.")

            # Flatten true and imputed values
            true_values = true_data.values.flatten()
            imputed_values = self.imputed_data[method].values.flatten()

            rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
            range_true = np.max(true_values) - np.min(true_values)
            nrmse = rmse / range_true
            print(f"Method: {method}")
            print(f"RMSE: {rmse:.4f}")
            print(f"NRMSE: {nrmse:.4f}")
            print()



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

    # Evaluate performance for each imputation method
    estimator.evaluate_performance(estimator.data)

    # Perform Moore-Penrose inverse checks
    estimator.moore_penrose_checks()

    # Add any additional steps or custom methods as required
