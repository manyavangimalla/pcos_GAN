from setuptools import setup, find_packages

setup(
    name="pcos_gans",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'opencv-python>=4.5.0',
        'albumentations>=1.0.0',
        'transformers>=4.15.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.62.0',
        'tensorboard>=2.7.0',
        'scipy>=1.7.0'
    ]
) 