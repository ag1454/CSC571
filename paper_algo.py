import numpy as np
from scipy.linalg import svd, pinv

# Function to generate 10% missing values
def generate_missing(sequence):
    # Calculate the number of missing values to be generated
    num_missing = round(len(sequence) * 0.1)
    
    # Randomly select positions in the sequence
    missing_positions = np.random.choice(len(sequence), num_missing, replace=False)
    
    # Replace the selected positions with None
    for pos in missing_positions:
        sequence[pos] = None
    
    return sequence

# Dynamic Local Least Squares Imputation Function
def DLLSimpute(G):
    # While there are still missing values in G
    while None in G:
        # Step 1: Sort each row by the number of missing values
        G.sort(key=lambda x: x is None)
        
        # Step 2: Find the first missing position in the first row
        missing_pos = G.index(None)
        
        # Step 3: Starting from the missing value position (i,j), the column j is scanned.
        # Increasing i, if the position (i,j) is a missing value, remove the whole row.
        G = [x for i, x in enumerate(G) if i != missing_pos]
        
        # Step 4: Separate the rest of the matrix into left and right matrices, selecting the largest matrix between these two.
        left_matrix = G[:missing_pos]
        right_matrix = G[missing_pos+1:]
        if len(left_matrix) > len(right_matrix):
            G = left_matrix
        else:
            G = right_matrix
    
    # Return the imputed matrix
    return G

# Singular Value Decomposition (SVD) Function
def svd_impute(G):
    # Perform SVD
    U, s, VT = svd(G, full_matrices=False)
    
    # Compute the pseudoinverse of G
    G_pinv = pinv(G)
    
    # Return the imputed matrix
    return G_pinv

# Usage
# Assuming 'sequence' is your sequence with the class
sequence = 'ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG'

# Create a dictionary to convert the sequence to numerical data
seq_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, None: None}

# Convert the sequence to numerical data
sequence_num = [seq_dict[i] for i in sequence]
seq_copy = sequence_num
# Generate 10% missing values
sequence_num = generate_missing(sequence_num)

# Perform the DLLSimpute, SVD, computation reduction, and Moore-Penrose pseudoinverse calculations
G_imputed = DLLSimpute(sequence_num)
G_imputed_svd = svd_impute(np.array(G_imputed).reshape(-1,1))


# Compare seq_copy with G_imputed and print %
match_count = sum([1 for i, j in zip(seq_copy, G_imputed) if i == j])
percentage = (match_count / len(seq_copy)) * 100
print(f"Matching percentage: {percentage}%")

# Convert seq_copy to numpy array
seq_copy = np.array(seq_copy)

# Ensure seq_copy and G_imputed have the same length
if len(seq_copy) > len(G_imputed):
    seq_copy = seq_copy[:len(G_imputed)]
else:
    G_imputed = G_imputed[:len(seq_copy)]


# calculate NRMSE with out the part to set the diff to 1 if its zero

nrmse = np.sqrt(np.mean((seq_copy - G_imputed)**2)) / (np.max(seq_copy) - np.min(seq_copy))
nrmse = np.sqrt(np.mean((np.array(seq_copy) - np.array(G_imputed))**2)) / (np.max(seq_copy) - np.min(seq_copy))
print(f"NRMSE: {nrmse}")

# Calculate NRMSE with the part to set the diff to 1 if its zero
nrmse_diff = (seq_copy - G_imputed)**2

if np.all(nrmse_diff == 0):
    nrmse_diff = 1
else:
    nrmse_diff

nrmse = np.sqrt(np.mean(nrmse_diff)) / (np.max(seq_copy) - np.min(seq_copy))
print(f"NRMSE: {nrmse}")
