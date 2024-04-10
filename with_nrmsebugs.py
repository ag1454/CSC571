import numpy as np
from scipy.linalg import svd, pinv
from sklearn.metrics import mean_squared_error

# Function to generate 10% missing values
def generate_missing(sequence):
    num_missing = round(len(sequence) * 0.10)
    missing_positions = np.random.choice(len(sequence), num_missing, replace=False)
    for pos in missing_positions:
        sequence[pos] = None
    return sequence, missing_positions

# Dynamic Local Least Squares Imputation Function
def DLLSimpute(G):
    while None in G:
        G.sort(key=lambda x: x is None)
        missing_pos = G.index(None)
        G = [x for i, x in enumerate(G) if i != missing_pos]
        left_matrix = G[:missing_pos]
        right_matrix = G[missing_pos+1:]
        if len(left_matrix) > len(right_matrix):
            G = left_matrix
        else:
            G = right_matrix
    return G

# Singular Value Decomposition (SVD) Function
def svd_impute(G):
    U, s, VT = svd(G, full_matrices=False)
    G_pinv = pinv(G)
    return G_pinv, U, s, VT

# Function to calculate NRMSE
def calculate_nrmse(true_values, predicted_values, missing_positions):
    # Flatten the predicted_values
    predicted_values = predicted_values.flatten()
    
    # Only consider the positions that correspond to the non-missing values
    true_values = [true_values[i] for i in range(len(true_values)) if i not in missing_positions]
    predicted_values = [predicted_values[i] for i in range(len(predicted_values)) if i not in missing_positions]
    
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(true_values) - np.min(true_values))
    return nrmse




# Assuming 'sequence' is your sequence with the class
sequence = 'ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG'

# Create a dictionary to convert the sequence to numerical data
seq_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, None: None}

# Convert the sequence to numerical data
sequence_num = [seq_dict[i] for i in sequence]

# Make a copy of the original sequence for later comparison
original_sequence = sequence_num.copy()

# Generate 10% missing values
sequence_num, missing_positions = generate_missing(sequence_num)

# Perform the DLLSimpute, SVD, computation reduction, and Moore-Penrose pseudoinverse calculations
G_imputed = DLLSimpute(sequence_num)
G_imputed_pinv, U, s, VT = svd_impute(np.array(G_imputed).reshape(-1,1))  # Reshape G_imputed_svd

# Calculate and print the NRMSE for the imputed values
# Calculate and print the NRMSE
nrmse = calculate_nrmse(original_sequence, G_imputed_pinv.flatten(), missing_positions)
print(f"The NRMSE of the imputation is: {nrmse}")


