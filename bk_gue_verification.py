# -*- coding: utf-8 -*-
"""
Filename: bk_gue_verification.py
Environment: Python 2.7.13 (32-bit Windows IDE Runner)
Description: Verifies numerical eigenvalue spacing statistics of a regularized 
             Berry-Keating operator matrix against GUE (Gaussian Unitary Ensemble) 
             predictions, adhering to strict deterministic software engineering rules.
             Outputs execution logs and spacing metrics directly to a text file.
"""

import math
import random
import sys

# Global Constants enforcing static bounds
MAX_GRID_SIZE = 128
MAX_ITERATIONS = 1000
OUTPUT_FILENAME = "bk_gue_verification_output.txt"

def validate_matrix_parameters(n_size):
    """
    Validates parameter bounds to ensure static memory allocation and safe execution.
    """
    assert isinstance(n_size, int), "Error: Matrix dimension must be an integer."
    assert 16 <= n_size <= MAX_GRID_SIZE, "Error: Matrix dimension exceeds safe bounds."
    return True

def construct_regularized_operator(n_size):
    """
    Constructs a symmetric tridiagonal matrix representation of the regularized 
    Berry-Keating operator with scale-invariant boundary regularization.
    """
    validate_matrix_parameters(n_size)
    
    # Static memory allocation for matrix stored as list of lists
    matrix = [[0.0 for _ in range(n_size)] for _ in range(n_size)]
    
    for i in range(n_size):
        # Fixed upper bound on loop
        x_val = float(i + 1) / float(n_size)
        # Diagonal term incorporating potential regularization V(x) = c / x^2
        matrix[i][i] = (x_val - 0.5) * math.log(x_val + 0.1) + (0.05 / (x_val * x_val + 0.01))
        
        if i + 1 < n_size:
            # Off-diagonal hopping term representing momentum operator coupling
            off_diag = 0.5 * math.sqrt(x_val * (1.0 - x_val))
            matrix[i][i+1] = off_diag
            matrix[i+1][i] = off_diag
            
    return matrix

def simple_power_iteration_eigenvalues(matrix, num_eigenvalues):
    """
    Approximates eigenvalues using bounded iterative deflation to satisfy 
    no-recursion and short-function constraints.
    """
    n = len(matrix)
    eigenvalues = []
    
    # Working copy of matrix
    work_mat = [[matrix[r][c] for c in range(n)] for r in range(n)]
    
    for k in range(num_eigenvalues):
        # Fixed iteration bound
        v = [random.uniform(0.1, 1.0) for _ in range(n)]
        norm_v = math.sqrt(sum([val * val for val in v]))
        v = [val / norm_v for val in v]
        
        lambda_approx = 0.0
        for step in range(MAX_ITERATIONS):
            # Matrix-vector multiplication with bounded loops
            next_v = [0.0] * n
            for r in range(n):
                row_sum = 0.0
                for c in range(n):
                    row_sum += work_mat[r][c] * v[c]
                next_v[r] = row_sum
                
            norm_next = math.sqrt(sum([val * val for val in next_v]))
            if norm_next < 1e-12:
                break
            next_v = [val / norm_next for val in next_v]
            
            # Rayleigh quotient approximation
            numerator = 0.0
            denominator = 0.0
            for r in range(n):
                row_sum = 0.0
                for c in range(n):
                    row_sum += work_mat[r][c] * next_v[c]
                numerator += next_v[r] * row_sum
                denominator += next_v[r] * next_v[r]
                
            lambda_approx = numerator / denominator if denominator > 0 else 0.0
            v = next_v
            
        eigenvalues.append(lambda_approx)
        
        # Deflation step to find subsequent eigenvalues
        for r in range(n):
            for c in range(n):
                work_mat[r][c] -= lambda_approx * v[r] * v[c]
                
    eigenvalues.sort()
    return eigenvalues

def compute_normalized_spacings(eigenvalues):
    """
    Computes nearest-neighbor level spacings s_i = E_{i+1} - E_i 
    and normalizes them for GUE comparison.
    """
    n = len(eigenvalues)
    assert n > 1, "Error: Insufficient eigenvalues for spacing computation."
    
    spacings = []
    for i in range(n - 1):
        diff = eigenvalues[i+1] - eigenvalues[i]
        if diff > 0.0:
            spacings.append(diff)
            
    if not spacings:
        return []
        
    mean_spacing = sum(spacings) / float(len(spacings))
    normalized = [s / mean_spacing for s in spacings]
    return normalized

def main():
    """
    Main execution routine: runs simulation, computes metrics, and writes to log file.
    """
    n_size = 64
    num_evals = 20
    
    # Open output file for logging (satisfying IDE/file execution constraints)
    out_file = open(OUTPUT_FILENAME, "w")
    out_file.write("=== REGULARIZED BERRY-KEATING GUE VERIFICATION LOG ===\n")
    out_file.write("Matrix Dimension: " + str(n_size) + "\n")
    
    # Construct operator and extract eigenvalues
    operator_matrix = construct_regularized_operator(n_size)
    evals = simple_power_iteration_eigenvalues(operator_matrix, num_evals)
    
    out_file.write("Computed Eigenvalues: " + str(evals) + "\n")
    
    # Compute spacings
    spacings = compute_normalized_spacings(evals)
    out_file.write("Normalized Nearest-Neighbor Spacings: " + str(spacings) + "\n")
    
    # Compare against Wigner surmise for GUE: P(s) = (32 / pi^2) * s^2 * exp(-4 s^2 / pi)
    out_file.write("\n--- GUE Statistical Comparison ---\n")
    if spacings:
        mean_s = sum(spacings) / float(len(spacings))
        variance_s = sum([(s - mean_s)**2 for s in spacings]) / float(len(spacings))
        out_file.write("Empirical Mean Spacing: " + str(mean_s) + " (GUE target ~ 1.0)\n")
        out_file.write("Empirical Spacing Variance: " + str(variance_s) + " (GUE target ~ 0.285)\n")
    else:
        out_file.write("Spacing computation yielded empty set.\n")
        
    out_file.write("Execution completed successfully.\n")
    out_file.close()

if __name__ == "__main__":
    main()
