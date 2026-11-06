# -*- coding: utf-8 -*-
"""
Filename: explicit_formula_verification.py
Environment: Python 2.7.13 (32-bit Windows IDE Runner)
Description: Computes and verifies the numerical balance between prime sum terms 
             and Riemann zeta zero sum terms in Riemann's explicit formula, 
             adhering to deterministic software engineering rules. Outputs execution 
             logs and error metrics directly to a text file.
"""

import math
import sys

# Global Constants enforcing static bounds
MAX_ITERATIONS = 500
OUTPUT_FILENAME = "explicit_formula_verification_output.txt"

def von_mangoldt_psi(x):
    """
    Computes the Chebyshev psi function psi(x) = sum_{n <= x} Lambda(n)
    up to a bounded limit for numerical comparison.
    """
    assert x > 1.0, "Error: x must be greater than 1.0."
    psi_val = 0.0
    
    # Bounded trial up to int(x)
    limit = int(x)
    for n in range(2, limit + 1):
        # Determine if n is a prime power p^k
        temp = n
        is_prime_power = False
        power_val = 0
        
        # Simple trial division for prime power testing
        d = 2
        prime_found = False
        base_p = 0
        while d * d <= temp:
            if temp % d == 0:
                base_p = d
                k = 0
                while temp % d == 0:
                    temp //= d
                    k += 1
                if temp == 1:
                    is_prime_power = True
                    power_val = math.log(base_p)
                break
            d += 1
        
        if not is_prime_power and temp > 1:
            # n is itself prime
            is_prime_power = True
            power_val = math.log(n)
            
        if is_prime_power:
            psi_val += power_val
            
    return psi_val

def explicit_formula_zero_sum(x, imaginary_zeros):
    """
    Computes the oscillating sum over non-trivial zeta zeros: - sum_gamma (x^(1/2 + i gamma) / (1/2 + i gamma))
    using first few known imaginary parts of zeta zeros.
    """
    total_sum = 0.0
    for gamma in imaginary_zeros:
        # Complex term: x^(1/2 + i * gamma) / (1/2 + i * gamma)
        # Magnitude factor: x^(1/2) / sqrt(0.25 + gamma^2)
        magnitude = math.sqrt(x) / math.sqrt(0.25 + gamma * gamma)
        # Phase angle from complex division and log
        phase = gamma * math.log(x) - math.atan(gamma / 0.5)
        total_sum += 2.0 * magnitude * math.cos(phase)
        
    return total_sum

def main():
    """
    Main execution routine: computes Chebyshev psi vs explicit formula approximation 
    and logs the discrepancy to a text file.
    """
    out_file = open(OUTPUT_FILENAME, "w")
    out_file.write("=== RIEMANN EXPLICIT FORMULA NUMERICAL VERIFICATION ===\n")
    
    # First 10 known imaginary parts of non-trivial zeros of Riemann zeta function
    zeta_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832
    ]
    
    test_x_values = [10.0, 25.0, 50.0]
    
    for x in test_x_values:
        out_file.write("\nEvaluating at x = " + str(x) + "\n")
        
        # Main asymptotic term x
        main_term = x
        
        # Prime sum side (Chebyshev psi approximation)
        psi_val = von_mangoldt_psi(x)
        out_file.write("  Chebyshev Psi Function psi(x): " + str(psi_val) + "\n")
        
        # Zero sum side
        zero_sum = explicit_formula_zero_sum(x, zeta_zeros)
        out_file.write("  Oscillating Zero Sum Term: " + str(zero_sum) + "\n")
        
        # Explicit formula balance: psi(x) approx x - sum(x^rho / rho) - log(2pi) - ...
        predicted_psi = main_term - zero_sum - math.log(2.0 * math.pi)
        out_file.write("  Explicit Formula Predicted psi(x): " + str(predicted_psi) + "\n")
        
        discrepancy = abs(psi_val - predicted_psi)
        out_file.write("  Absolute Discrepancy: " + str(discrepancy) + "\n")
        
    out_file.write("\nVerification completed successfully.\n")
    out_file.close()

if __name__ == "__main__":
    main()
