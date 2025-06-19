# Polycystic Ovary Syndrome Recognition Using GANs with Few-Shot Learning

## Overview
This project aims to develop an advanced system for the recognition of Polycystic Ovary Syndrome (PCOS) by leveraging Generative Adversarial Networks (GANs) and Few-Shot Learning. The pipeline integrates multi-modal data, including ultrasound images, hormonal data, and clinical records, and utilizes state-of-the-art deep learning and NLP techniques for robust PCOS detection.

## Features
- **Data Preprocessing:** Handles ultrasound images, hormonal data, and clinical records.
- **GAN Module:** Generates synthetic data to augment limited datasets.
- **Few-Shot Learning:** Employs MAML-based models for learning from few examples.
- **NLP Pipeline:** Uses BioBERT for extracting features from clinical text.
- **Evaluation:** Calculates accuracy, precision, recall, and AUC-ROC.
- **Utilities:** Includes helpers for data loading, visualization, and experiment management.

## Methodology
1. **Data Preprocessing:** Cleans and formats multi-modal data.
2. **GAN Training:** Trains GANs to generate synthetic samples for data augmentation.
3. **Few-Shot Learning:** Applies MAML to enable learning from limited labeled data.
4. **NLP Analysis:** Extracts features from clinical records using BioBERT.
5. **Evaluation:** Assesses model performance using standard metrics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/manyavangimalla/pcos_GAN.git
   cd pcos_GAN
   ```
2. Set up the environment (recommended: Anaconda):
   ```bash
   conda create -n pcos_env python=3.9
   conda activate pcos_env
   conda install pytorch pandas scikit-learn opencv
   pip install albumentations tensorboard
   pip install -r PCOS_GANs/requirements.txt
   ```

## Usage
- **Preprocess Data:**
  ```bash
  python PCOS_GANs/run.py --stage preprocess
  ```
- **Train GAN:**
  ```bash
  python PCOS_GANs/run.py --stage train_gan
  ```
- **Train Few-Shot Model:**
  ```bash
  python PCOS_GANs/run.py --stage train_few_shot
  ```
- **Evaluate:**
  ```bash
  python PCOS_GANs/run.py --stage evaluate
  ```

## Project Structure
```
PCOS_GANs/
  ├── config.yaml
  ├── data/
  ├── experiments/
  ├── main.py
  ├── notebooks/
  ├── requirements.txt
  ├── results/
  ├── run.py
  ├── setup.py
  └── src/
      ├── evaluation/
      ├── few_shot/
      ├── gan/
      ├── models/
      ├── nlp/
      ├── preprocessing/
      └── utils/
```

## Citation
If you use this code or ideas from this project, please cite:
```
@misc{pcos_gan,
  author = {Manya Vangimalla},
  title = {Polycystic Ovary Syndrome Recognition Using GANs with Few-Shot Learning},
  year = {2024},
  howpublished = {\url{https://github.com/manyavangimalla/pcos_GAN}}
}
```

## License
This project is licensed under the MIT License.

## Contact
For questions or collaborations, please contact [Manya Vangimalla](mailto:your.email@example.com). 