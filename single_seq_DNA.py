"""
I kept running into issue with multiple sequences, so I decided to focus on a single sequence.
This works and I am able to get the NRMSE for the KNN, LLS, and DLL imputation methods. 
as well as print the difference table for each method.

I got it to run on two sequences and results kinda of matches when 10% is missing from the sequence.

"""


import numpy as np
from sklearn.metrics import mean_squared_error

def generate_missing_sequence(sequence, missing_percentage):
    num_missing = int(len(sequence) * missing_percentage)
    missing_indices = np.random.choice(len(sequence), num_missing, replace=False)
    missing_sequence = list(sequence)
    for idx in missing_indices:
        missing_sequence[idx] = 'N'
    return ''.join(missing_sequence)

def KNNimpute_single_sequence(sequence, k):
    sequence_array = list(sequence)
    
    for i, nucleotide in enumerate(sequence_array):
        if nucleotide == 'N':  # If the value is missing
            local_neighborhood = sequence_array[max(0, i-k):min(len(sequence_array), i+k+1)]
            local_neighborhood = [x for x in local_neighborhood if x != 'N']  # Remove missing values
            
            if len(local_neighborhood) == 0:  # If no data points available
                sequence_array[i] = 'A'  # Impute with a default nucleotide (e.g., 'A')
            else:
                # Impute with the most common nucleotide in the local neighborhood
                sequence_array[i] = max(set(local_neighborhood), key=local_neighborhood.count)
    
    return ''.join(sequence_array)

def LLSimpute_single_sequence(sequence, neighborhood_size):
    sequence_array = list(sequence)
    
    for i, nucleotide in enumerate(sequence_array):
        if nucleotide == 'N':  # If the value is missing
            local_neighborhood = sequence_array[max(0, i-neighborhood_size):min(len(sequence_array), i+neighborhood_size+1)]
            local_neighborhood = [x for x in local_neighborhood if x != 'N']  # Remove missing values
            
            if len(local_neighborhood) == 0:  # If no data points available
                sequence_array[i] = 'A'  # Impute with a default nucleotide (e.g., 'A')
            else:
                # Impute with the most common nucleotide in the local neighborhood
                sequence_array[i] = max(set(local_neighborhood), key=local_neighborhood.count)
    
    return ''.join(sequence_array)

def DLLSimpute_single_sequence(sequence, max_neighborhood_size):
    sequence_array = list(sequence)
    
    for i, nucleotide in enumerate(sequence_array):
        if nucleotide == 'N':  # If the value is missing
            neighborhood_size = 1  # Start with a small neighborhood size
            
            while neighborhood_size <= max_neighborhood_size:
                # Expand the local neighborhood
                local_neighborhood = sequence_array[max(0, i-neighborhood_size):min(len(sequence_array), i+neighborhood_size+1)]
                local_neighborhood = [x for x in local_neighborhood if x != 'N']  # Remove missing values
                
                if len(local_neighborhood) == 0:  # If no data points available
                    neighborhood_size += 1
                else:
                    # Impute with the most common nucleotide in the local neighborhood
                    sequence_array[i] = max(set(local_neighborhood), key=local_neighborhood.count)
                    break  # Exit the loop once the missing value is imputed
            
            if neighborhood_size > max_neighborhood_size:  # If no suitable neighborhood found
                sequence_array[i] = 'A'  # Impute with a default nucleotide (e.g., 'A')
    
    return ''.join(sequence_array)

def calculate_nrmse(original_sequence, imputed_sequence):
    return np.sqrt(mean_squared_error([ord(x) for x in original_sequence], [ord(x) for x in imputed_sequence])) / (np.max([ord(x) for x in original_sequence]) - np.min([ord(x) for x in original_sequence]))

def print_difference_table(original_sequence, imputed_sequence):
    print("Position | Original | Imputed | Difference")
    print("-------------------------------------------")
    for i, (original_nucleotide, imputed_nucleotide) in enumerate(zip(original_sequence, imputed_sequence)):
        if original_nucleotide != imputed_nucleotide:
            print(f"{i+1:8} | {original_nucleotide:8} | {imputed_nucleotide:7} | {'Different':10}")
        else:
            print(f"{i+1:8} | {original_nucleotide:8} | {imputed_nucleotide:7} | {'Same':10}")

# Example usage
sequences = [
    "ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG",
    "ATGCTAACCTAAAGCACGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG"
]
missing_percentage = 0.1  # 10% missing values

for sequence in sequences:
    # Generate missing sequence
    missing_sequence = generate_missing_sequence(sequence, missing_percentage)

    # KNN imputation
    knn_imputed_sequence = KNNimpute_single_sequence(missing_sequence, k=5)

    # LLSimpute
    lls_imputed_sequence = LLSimpute_single_sequence(missing_sequence, neighborhood_size=5)

    # DLLSimpute
    dll_imputed_sequence = DLLSimpute_single_sequence(missing_sequence, max_neighborhood_size=5)

    # Calculate NRMSE
    knn_nrmse = calculate_nrmse(sequence, knn_imputed_sequence)
    lls_nrmse = calculate_nrmse(sequence, lls_imputed_sequence)
    dll_nrmse = calculate_nrmse(sequence, dll_imputed_sequence)

    # Print NRMSE
    print(f"\nNRMSE for sequence:\n{sequence}")
    print(f"KNN Imputation: {knn_nrmse}")
    print(f"LLSimpute: {lls_nrmse}")
    print(f"DLLSimpute: {dll_nrmse}")
    # Print difference table - KNN
    print("\nDifference Table for KNN Imputation:")
    print_difference_table(sequence, knn_imputed_sequence)
    num_different_knn = sum(1 for original_nucleotide, imputed_nucleotide in zip(sequence, knn_imputed_sequence) if original_nucleotide != imputed_nucleotide)
    num_same_knn = len(sequence) - num_different_knn
    print(f"Number of Different: {num_different_knn}")
    print(f"Number of Same: {num_same_knn}")

    # Print difference table - LLS
    print("\nDifference Table for LLSimpute:")
    print_difference_table(sequence, lls_imputed_sequence)
    num_different_lls = sum(1 for original_nucleotide, imputed_nucleotide in zip(sequence, lls_imputed_sequence) if original_nucleotide != imputed_nucleotide)
    num_same_lls = len(sequence) - num_different_lls
    print(f"Number of Different: {num_different_lls}")
    print(f"Number of Same: {num_same_lls}")

    # Print difference table - DLLS
    print("\nDifference Table for DLLSimpute:")
    print_difference_table(sequence, dll_imputed_sequence)
    num_different_dll = sum(1 for original_nucleotide, imputed_nucleotide in zip(sequence, dll_imputed_sequence) if original_nucleotide != imputed_nucleotide)
    num_same_dll = len(sequence) - num_different_dll
    print(f"Number of Different: {num_different_dll}")
    print(f"Number of Same: {num_same_dll}")
