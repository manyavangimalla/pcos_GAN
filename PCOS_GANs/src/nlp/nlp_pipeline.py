"""
NLP pipeline for processing clinical records using BioBERT.
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import os

class ClinicalRecordDataset(Dataset):
    """Dataset class for clinical records."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        """
        Initialize dataset.
        
        Args:
            texts (List[str]): List of clinical record texts
            labels (List[int]): List of labels
            tokenizer: Tokenizer for text processing
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ClinicalNLPPipeline:
    def __init__(self, config: Dict):
        """
        Initialize NLP pipeline.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'dmis-lab/biobert-v1.1',
            num_labels=config['num_classes']
        ).to(self.device)
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(config['output_dir'], 'logs'),
            logging_steps=100,
        )
    
    def prepare_data(self,
                    texts: List[str],
                    labels: List[int]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare datasets for training.
        
        Args:
            texts (List[str]): List of clinical record texts
            labels (List[int]): List of labels
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets
        """
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset = ClinicalRecordDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = ClinicalRecordDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = ClinicalRecordDataset(test_texts, test_labels, self.tokenizer)
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Train the model.
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
        """
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(os.path.join(self.config['output_dir'], 'best_model'))
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts (List[str]): List of clinical record texts
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        self.model.eval()
        predictions = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.vstack(predictions)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from texts using BioBERT.
        
        Args:
            texts (List[str]): List of clinical record texts
            
        Returns:
            np.ndarray: Extracted features
        """
        self.model.eval()
        features = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token representation as features
                features.append(outputs.hidden_states[-1][:, 0].cpu().numpy())
        
        return np.vstack(features)

def main():
    """Main function for NLP pipeline."""
    config = {
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 5,
        'output_dir': '../results/nlp'
    }
    
    pipeline = ClinicalNLPPipeline(config)
    
    # Example usage:
    # 1. Prepare data
    # train_dataset, val_dataset, test_dataset = pipeline.prepare_data(texts, labels)
    
    # 2. Train model
    # pipeline.train(train_dataset, val_dataset)
    
    # 3. Make predictions
    # predictions = pipeline.predict(new_texts)
    
    print("NLP pipeline completed!")

if __name__ == "__main__":
    main() 