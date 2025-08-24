# Quick test script to verify fraud detection
import pandas as pd
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test data with clear fraud indicators
test_data = {
    'Time': ['12:30', '23:45', '08:15', '02:30'],  # Two late night transactions
    'Amount': [50.00, 2500.00, 75.25, 1800.00],  # Two high amounts
    'V1': [0.1, 2.1, 0.2, -3.1],  # Last one has extreme value
    'V2': [0.2, 3.2, -0.1, 4.2],
    'V3': [-0.1, -2.8, 0.3, -3.8],
    'V4': [0.3, 4.3, 0.1, 3.9],
    'V5': [-0.2, -3.2, -0.2, -4.1],
    'V6': [0.1, 2.1, 0.4, 3.2],
    'V7': [0.4, 3.4, 0.1, -3.4],
    'V8': [-0.1, -2.9, -0.3, 4.1],
    'V9': [0.2, 3.2, 0.2, -3.2],
    'V10': [0.1, 2.8, 0.1, 3.8],
    'V11': [-0.3, -4.1, 0.3, 4.1],
    'V12': [0.2, 3.2, -0.2, -3.9],
    'V13': [0.1, 2.9, 0.1, 3.2],
    'V14': [-0.2, -3.2, 0.4, -4.1],
    'V15': [0.3, 4.1, -0.1, 3.8],
    'V16': [0.1, 2.8, 0.2, -3.2],
    'V17': [-0.1, -2.9, 0.3, 4.1],
    'V18': [0.2, 3.2, 0.1, -3.9],
    'V19': [0.1, 2.1, -0.2, 3.2],
    'V20': [0.3, 4.3, 0.4, -4.1],
    'V21': [-0.2, -3.2, 0.1, 3.8],
    'V22': [0.1, 2.8, 0.3, 3.2],
    'V23': [0.2, 3.2, -0.2, -4.1],
    'V24': [-0.1, -2.9, 0.1, 3.9],
    'V25': [0.3, 4.1, 0.4, -3.2],
    'V26': [0.1, 2.8, 0.2, 4.1],
    'V27': [0.2, 3.2, -0.1, -3.8],
    'V28': [0.0, 0.0, 0.0, 0.0]
}

df = pd.DataFrame(test_data)
print("Test data created:")
print("Transaction 1: Normal daytime, small amount")
print("Transaction 2: Late night (23:45), high amount ($2500), extreme V values")
print("Transaction 3: Morning, normal amount")
print("Transaction 4: Very late night (02:30), high amount ($1800), extreme V values")
print("\nExpected: Transactions 2 and 4 should be flagged as fraud")
print(f"Data shape: {df.shape}")
