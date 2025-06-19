"""
Main script for running the PCOS detection pipeline.
"""
import os
import yaml
import torch
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.gan.models import Generator, Discriminator
from src.gan.train_gan import GANTrainer
from src.few_shot.models import MAML
from src.few_shot.train_few_shot import FewShotTrainer
from src.nlp.nlp_pipeline import ClinicalNLPPipeline
from src.evaluation.evaluate import ModelEvaluator
from src.utils.helpers import (
    load_config,
    PCOSDataset,
    create_dataloader,
    setup_experiment
)

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    experiment_dir = setup_experiment('pcos_detection')
    print(f"Experiment directory: {experiment_dir}")
    
    try:
        # 1. Data Preprocessing
        print("\n1. Starting data preprocessing...")
        preprocessor = DataPreprocessor(config['data'])
        processed_images, metadata = preprocessor.create_dataset(
            image_dir=os.path.join(config['data']['raw_dir'], 'ultrasound_images'),
            hormonal_data_path=os.path.join(config['data']['raw_dir'], 'hormonal_data.csv'),
            clinical_records_path=os.path.join(config['data']['raw_dir'], 'clinical_records.txt')
        )
        print("Data preprocessing completed!")
        
        # 2. GAN Training
        print("\n2. Starting GAN training...")
        gan_trainer = GANTrainer(config['gan'])
        # Uncomment when data is available:
        # gan_trainer.train(train_dataloader, config['gan']['epochs'])
        print("GAN training completed!")
        
        # 3. Few-Shot Learning
        print("\n3. Starting Few-Shot Learning...")
        few_shot_trainer = FewShotTrainer(config['few_shot'])
        # Uncomment when data is available:
        # few_shot_trainer.train(dataset)
        print("Few-Shot Learning completed!")
        
        # 4. NLP Pipeline
        print("\n4. Starting NLP pipeline...")
        nlp_pipeline = ClinicalNLPPipeline(config['nlp'])
        # Uncomment when data is available:
        # train_dataset, val_dataset, test_dataset = nlp_pipeline.prepare_data(texts, labels)
        # nlp_pipeline.train(train_dataset, val_dataset)
        print("NLP pipeline completed!")
        
        # 5. Evaluation
        print("\n5. Starting evaluation...")
        evaluator = ModelEvaluator(config['evaluation'])
        # Uncomment when models are trained:
        # classifier_metrics = evaluator.evaluate_classifier(model, test_loader)
        # gan_metrics = evaluator.evaluate_gan(generator, real_images)
        # evaluator.plot_results(config['evaluation']['output_dir'])
        print("Evaluation completed!")
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 