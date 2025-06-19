# Polycystic Ovary Syndrome Recognition Using GANs with Few-Shot Learning

## Abstract
Polycystic Ovary Syndrome (PCOS) is a severe endocrine disorder affecting women of reproductive age and leading to complications such as infertility, metabolic syndrome and increased risk of type 2 diabetes. Type 2 diabetes is influenced by a combination of modifiable and non-modifiable risk factors. Modifiable risk factors for type 2 diabetes include excess body weight and smoking. Non-modifiable risk factors include age with the risk increasing around age 45 and rising after age 65. Early recognition is crucial for effective management and treatment. Generative Adversarial Networks are employed to generate synthetic ultrasound images, augmenting the dataset and mitigating data scarcity. Medical datasets suffer from limited sample sizes due to privacy concerns and the rarity of certain conditions. GANs improve image quality by reducing noise and correcting artifacts, leading to clear and more accurate diagnostic images. The Few-Shot learning framework will enable the model to achieve PCOS detection with minimal labeled data, facilitating early diagnosis and effective management of the condition. The proposed framework has the potential to significantly improve diagnostic precision, streamline healthcare workflows and facilitate better patient outcomes through timely and effective management of this condition. Considering various machine learning models, such as a convolutional neural network (CNN), Logistic Regression, Decision Tree, Naive Bayes etc., this paper aims to provide accurate and scalable solution for early detection of PCOS.

## Introduction
Polycystic Ovary Syndrome (PCOS) is a complex endocrine disorder that affects millions of women of reproductive age globally. Women with PCOS often experience higher rates of anxiety and depression, potentially due to hormonal effects and body image concerns. It is characterized by a range of symptoms including irregular menstrual cycles, excessive androgen levels, and polycystic ovaries. Hormones are chemical messengers produced by endocrine glands that regulate various physiological processes including metabolism, growth, reproduction, and mood. Prolonged high cortisol levels can interfere with the production and function of other hormones, leading to symptoms like growing hair on the face, chest, back, and buttocks, weight gain, mood swings, and disrupted sleep patterns. Generative Adversarial Networks (GANs) are being used in healthcare applications, including the diagnosis and study of Polycystic Ovary Syndrome (PCOS). GANs consist of two neural networks: the generator and the discriminator, which compete in a game-theoretic framework. Machine Learning algorithms can integrate and analyze heterogeneous data sources, such as hormonal levels, ultrasound images, and patient history, to identify early signs of PCOS. Natural Language Processing (NLP) tools can extract relevant patterns from patient records and combine them with clinical markers for diagnosis.

## Goal and Unique Contribution
The primary goal of this project is to develop an innovative framework which is Generative Adversarial Networks (GANs) and Few-Shot Learning for early detection and diagnosis of Polycystic Ovary Syndrome (PCOS). The framework aims to address critical challenges in PCOS diagnosis, such as data scarcity, variability in symptoms, and the need for precise, personalized healthcare solutions. GANs to generate synthetic ultrasound images, the project seeks to augment limited datasets, enhance model training, and improve diagnostic accuracy. The integration of Few-Shot Learning enables effective PCOS detection even with minimal labeled data, reducing the reliance on large, annotated datasets that are often challenging to obtain in medical research.

The unique contribution of this project aims to combine the strengths of GANs and Few-Shot Learning to create a scalable and robust diagnostic tool. Unlike traditional methods that depend on extensive datasets and manual feature extraction, this framework employs state-of-the-art machine learning techniques, such as Convolutional Neural Networks (CNNs) and Natural Language Processing (NLP), to analyze diverse data sources like ultrasound images, clinical records, and hormonal profiles. By streamlining diagnostic workflows, the proposed framework offers a precise, cost-effective, and accessible solution with the potential to improve patient outcomes through diagnosis and personalized care.

## Methodology (Summary)
- **Data Collection:** Gather ultrasound images, hormonal profiles, and clinical records.
- **Preprocessing:** Clean, normalize, and format data for ML models.
- **GANs:** Generate synthetic ultrasound images to augment the dataset and improve image quality.
- **Few-Shot Learning:** Use frameworks like MAML to enable learning from minimal labeled data.
- **Model Training:** Train and compare CNNs, SVMs, Logistic Regression, Decision Trees, and NLP models.
- **Evaluation:** Use metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
- **Validation:** Compare with baseline methods and validate on real-world clinical data.

For more details, see the flowchart and documentation in the repository.

## Project Structure

```
PCOS_GANs/
├── data/
│   ├── processed/      # Preprocessed data
│   ├── raw/           # Raw input data
│   └── synthetic/     # GAN-generated synthetic data
├── src/
│   ├── preprocessing/ # Data preprocessing modules
│   ├── gan/          # GAN implementation
│   ├── few_shot/     # Few-shot learning implementation
│   ├── nlp/          # Clinical text processing
│   ├── evaluation/   # Model evaluation
│   └── utils/        # Helper functions
├── results/          # Training results and metrics
└── notebooks/        # Jupyter notebooks for analysis
```

## Setup Instructions

### Prerequisites

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Environment Setup

1. Create and activate a new conda environment:
```bash
conda create -n pcos_env python=3.9
conda activate pcos_env
```

2. Install PyTorch and other dependencies:
```bash
# Install PyTorch with CPU support
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
conda install numpy pandas scikit-learn matplotlib seaborn opencv
pip install albumentations transformers pyyaml tqdm tensorboard
```

## Running the Project

1. Prepare your data:
   - Place ultrasound images in `data/raw/ultrasound_images/`
   - Place hormonal data CSV in `data/raw/hormonal_data.csv`
   - Place clinical records in `data/raw/clinical_records.txt`

2. Run the preprocessing pipeline:
```bash
python run.py
```

3. The processed data will be saved in the `data/processed/` directory.

## Components

1. **Data Preprocessing**
   - Image preprocessing for ultrasound scans
   - Hormonal data normalization
   - Clinical text processing

2. **GAN Training**
   - Generates synthetic ultrasound images
   - Helps augment the training dataset

3. **Few-Shot Learning**
   - MAML-based implementation
   - Efficient learning from limited data

4. **NLP Pipeline**
   - Processes clinical records
   - Extracts relevant features from text data

5. **Evaluation**
   - Performance metrics calculation
   - Results visualization

## Configuration

The project uses a YAML configuration file (`config.yaml`) to manage parameters:
- Data paths and preprocessing settings
- GAN architecture and training parameters
- Few-shot learning parameters
- Evaluation metrics

## Results

Training results, including:
- Generated images
- Model checkpoints
- Evaluation metrics
- Visualizations

are saved in the `results/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 