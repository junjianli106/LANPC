# Deep Learning Radiopathomic Model for Locally Advanced Nasopharyngeal Carcinoma

This repository contains the code and resources for the project titled **Deep Learning Radiopathomic Model Based on Pretreatment MRI and Whole Slide Images for Predicting Overall Survival in Locally Advanced Nasopharyngeal Carcinoma**. The project focuses on developing a multimodal fusion model for prognosis prediction using Whole Slide Imaging (WSI) and Magnetic Resonance Imaging (MRI) data. 

## Key Features
1. Pretrained Model  
   - We used a ResNet-50 model pretrained on ImageNet (PyTorch implementation).  
   - The model weights and implementation details are publicly available at [Hugging Face](https://huggingface.co/microsoft/resnet-50).  

2. Multimodal Fusion  
   - a multimodal feature fusion module integrated the WSI-level and MRI-level features. To achieve this, we first concatenated the WSI-level and MRI-level features, combining the strengths of both modalities. We then employed an MLP network to learn the interactions between these two types of features, enabling the model to leverage the complementary information from WSIs and MRI.

3. Prognosis Definition  
   - The model employs a Cox proportional hazards layer to handle survival data directly.  
   - Risk scores were dichotomized at the median to stratify patients into high-risk and low-risk groups.  


## Repository Structure
```
project_repository/
├── datasets/               # Directory for datasets (not included due to privacy concerns)
├── models/                 # Pretrained models and custom architecture implementations
├── splits/                 # Predefined data splits for training, validation, and testing
├── train.py                # Main script for training the model
├── README.md               # This file
└── requirements.txt        # List of dependencies
```

## Usage Instructions
1. Install Dependencies  
   Install the required dependencies using the following command:  
   ```bash
   pip install -r requirements.txt
   ```

2. Preprocessing  
   - For pathology data, we use CLAM to extract features and apply stain normalization.  
   - For MRI data, we use Pyradiomics to extract features.  

3. Training  
   Train the model using the provided training script:  
   ```bash
   python train.py --stage='train' --config='HPCH/LANPC.yaml' --gpus=0
   ```

We hope this repository will aid in reproducing our results and further research in this field. Thank you for your interest in our work!

For more details about the ResNet-50 model, visit: [ResNet-50 Model on Hugging Face](https://huggingface.co/microsoft/resnet-50)
