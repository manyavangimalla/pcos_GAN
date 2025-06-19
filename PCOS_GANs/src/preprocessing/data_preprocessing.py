"""
Data preprocessing utilities for PCOS project.
Handles ultrasound images, clinical records, and hormonal data.
"""
import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import albumentations as A

class DataPreprocessor:
    def __init__(self, config: Dict):
        """
        Initialize data preprocessor with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.image_size = config.get('image_size', (256, 256))
        self.image_augmentation = self._get_augmentation_pipeline()
        self.scaler = StandardScaler()
        
    def _get_augmentation_pipeline(self) -> A.Compose:
        """
        Create an image augmentation pipeline using albumentations.
        
        Returns:
            A.Compose: Augmentation pipeline
        """
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess_ultrasound_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess ultrasound image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.image_augmentation(image=image)
        return augmented['image']
    
    def preprocess_hormonal_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess hormonal profile data.
        
        Args:
            data (pd.DataFrame): Raw hormonal data
            
        Returns:
            pd.DataFrame: Preprocessed hormonal data
        """
        # Drop patient_id as it's not needed for training
        if 'patient_id' in data.columns:
            data = data.drop('patient_id', axis=1)
        
        # Separate features and target
        X = data.drop('pcos', axis=1)
        y = data['pcos']
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Add target back
        X_scaled['pcos'] = y
        
        return X_scaled
    
    def preprocess_clinical_records(self, records: List[str]) -> List[str]:
        """
        Preprocess clinical records text data.
        
        Args:
            records (List[str]): List of clinical records
            
        Returns:
            List[str]: Preprocessed clinical records
        """
        processed_records = []
        for record in records:
            # Remove patient ID prefix
            record = record.split(':', 1)[1].strip()
            # Convert to lowercase
            record = record.lower()
            # Remove periods
            record = record.replace('.', '')
            processed_records.append(record + '\n')
        
        return processed_records
    
    def create_dataset(self, 
                      image_dir: str,
                      hormonal_data_path: str,
                      clinical_records_path: str) -> Tuple[Dict, pd.DataFrame]:
        """
        Create complete dataset by combining all data sources.
        
        Args:
            image_dir (str): Directory containing ultrasound images
            hormonal_data_path (str): Path to hormonal data CSV
            clinical_records_path (str): Path to clinical records file
            
        Returns:
            Tuple[Dict, pd.DataFrame]: Processed images and combined metadata
        """
        # Process images
        processed_images = {}
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                processed_images[img_name] = self.preprocess_ultrasound_image(img_path)
        
        # Process hormonal data
        hormonal_data = pd.read_csv(hormonal_data_path)
        processed_hormonal = self.preprocess_hormonal_data(hormonal_data)
        
        # Process clinical records
        with open(clinical_records_path, 'r') as f:
            clinical_records = f.readlines()
        processed_records = self.preprocess_clinical_records(clinical_records)
        
        # Combine metadata
        metadata = processed_hormonal.copy()
        metadata['clinical_record'] = processed_records
        
        return processed_images, metadata

def main():
    """Main function for data preprocessing pipeline."""
    config = {
        'image_size': (256, 256),
        'data_dir': '../data',
        'raw_dir': '../data/raw',
        'processed_dir': '../data/processed'
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Example usage
    try:
        processed_images, metadata = preprocessor.create_dataset(
            image_dir=os.path.join(config['raw_dir'], 'ultrasound_images'),
            hormonal_data_path=os.path.join(config['raw_dir'], 'hormonal_data.csv'),
            clinical_records_path=os.path.join(config['raw_dir'], 'clinical_records.txt')
        )
        print("Data preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main() 