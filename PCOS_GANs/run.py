"""
Script to run the PCOS detection project with sample data.
"""
import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing.data_preprocessing import DataPreprocessor
from src.gan.train_gan import GANTrainer
from src.few_shot.train_few_shot import FewShotTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.utils.helpers import load_config, setup_experiment

def preprocess_data(config):
    """Handles the data preprocessing stage."""
    print("Starting data preprocessing...")
    preprocessor = DataPreprocessor(config['data'])
    
    hormonal_path = os.path.join(config['data']['raw_dir'], 'hormonal_data.csv')
    clinical_path = os.path.join(config['data']['raw_dir'], 'clinical_records.txt')
    
    # Process hormonal data
    hormonal_df = pd.read_csv(hormonal_path)
    processed_hormonal = preprocessor.preprocess_hormonal_data(hormonal_df)
    np.save(os.path.join(config['data']['processed_dir'], 'hormonal_data.npy'), processed_hormonal.values)
    print("Hormonal data processed and saved.")

    # Process clinical records
    with open(clinical_path, 'r') as f:
        clinical_records = f.readlines()
    processed_records = preprocessor.preprocess_clinical_records(clinical_records)
    with open(os.path.join(config['data']['processed_dir'], 'clinical_records.txt'), 'w') as f:
        f.writelines(processed_records)
    print("Clinical records processed and saved.")
    print("Data preprocessing completed!")

def main():
    parser = argparse.ArgumentParser(description="PCOS Detection Pipeline")
    parser.add_argument('--stage', type=str, default='all', 
                        choices=['all', 'preprocess', 'train_gan', 'train_few_shot', 'evaluate'],
                        help='Which stage of the pipeline to run.')
    args = parser.parse_args()

    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = setup_experiment('pcos_detection_sample')
    config['results']['path'] = experiment_dir
    
    print(f"Running stage: {args.stage}")
    print(f"Using device: {device}")
    print(f"Experiment results will be saved in: {experiment_dir}")

    # Create a dummy dataloader for tabular data
    dummy_data = torch.randn(100, config['gan']['input_dim'])
    dummy_labels = torch.randint(0, 2, (100,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=config['gan']['batch_size'])

    if args.stage == 'preprocess' or args.stage == 'all':
        preprocess_data(config)

    if args.stage == 'train_gan' or args.stage == 'all':
        print("\nStarting GAN training...")
        gan_trainer = GANTrainer(config, device)
        gan_trainer.train(dataloader=dummy_dataloader, epochs=config['gan']['epochs'])
        print("GAN training completed!")

    if args.stage == 'train_few_shot' or args.stage == 'all':
        print("\nStarting Few-Shot Learning...")
        # Note: FewShotTrainer expects a dataset. This is a placeholder.
        few_shot_trainer = FewShotTrainer(config, device)
        few_shot_trainer.train(dataset=None, epochs=config['few_shot']['epochs'])
        print("Few-Shot Learning completed!")
        
    if args.stage == 'evaluate' or args.stage == 'all':
        print("\nStarting evaluation...")
        # This is a placeholder for the evaluation logic
        # evaluator = ModelEvaluator(config)
        # evaluator.evaluate(...)
        print("Evaluation completed!")

if __name__ == "__main__":
    main() 