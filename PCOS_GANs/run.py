"""
Script to run the PCOS detection project with sample data.
"""
import os
import torch
import numpy as np
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.utils.helpers import load_config, setup_experiment
import pandas as pd

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Set up experiment
    experiment_dir = setup_experiment('pcos_detection_sample')
    print(f"Experiment directory: {experiment_dir}")
    
    # Check data files
    required_files = [
        os.path.join('data', 'raw', 'hormonal_data.csv'),
        os.path.join('data', 'raw', 'clinical_records.txt')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found!")
            return
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config['data'])
        
        # Process data
        print("\nProcessing data...")
        processed_data = preprocessor.preprocess_hormonal_data(
            pd.read_csv(os.path.join('data', 'raw', 'hormonal_data.csv'))
        )
        print("Hormonal data processed!")
        
        # Process clinical records
        with open(os.path.join('data', 'raw', 'clinical_records.txt'), 'r') as f:
            clinical_records = f.readlines()
        processed_records = preprocessor.preprocess_clinical_records(clinical_records)
        print("Clinical records processed!")
        
        # Save processed data
        np.save(
            os.path.join('data', 'processed', 'hormonal_data.npy'),
            processed_data.values
        )
        
        with open(os.path.join('data', 'processed', 'clinical_records.txt'), 'w') as f:
            f.writelines(processed_records)
        
        print("\nData processing completed successfully!")
        print(f"Processed data saved in: {os.path.join('data', 'processed')}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 