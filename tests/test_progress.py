#!/usr/bin/env python3
"""Test script to verify tqdm progress bars are working"""

import time
from tqdm import tqdm

def test_progress_bar():
    """Test basic progress bar functionality"""
    print("Testing basic progress bar...")
    
    # Create a simple progress bar
    pbar = tqdm(total=100, desc="Test Progress", unit="items")
    
    # Update the progress bar
    for i in range(100):
        time.sleep(0.05)  # Simulate work
        pbar.update(1)
    
    pbar.close()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_progress_bar()