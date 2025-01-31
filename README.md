# Machine-Learning-Project
Comparison of Two Supervised Learning Methods for the Classification of  Jellyfish Images
# Jellyfish Image Classification: CNN vs. Transfer Learning
## Overview
This project compares two supervised learning methods—Convolutional Neural Networks (CNN) and Transfer Learning using the VGG16 architecture—for classifying six different species of jellyfish. The goal is to evaluate the effectiveness of these models based on accuracy, validation performance, and generalizability.

## Dataset
The dataset is sourced from Kaggle and consists of **979 images** categorized into six species:
- Moon jellyfish (*Aurelia aurita*)
- Barrel jellyfish (*Rhizostoma pulmo*)
- Blue jellyfish (*Cyanea lamarckii*)
- Compass jellyfish (*Chrysaora hysoscella*)
- Lion’s mane jellyfish (*Cyanea capillata*)
- Mauve stinger (*Pelagia noctiluca*)

The dataset is split into:
- **Training set**: 900 images
- **Validation set**: 39 images
- **Test set**: 40 images

Each image has dimensions **224 × 224 × 3 (RGB)**.

## Methods
### 1. **Convolutional Neural Networks (CNN)**
   - Custom CNN models were trained using multiple architectures.
   - Various strategies were tested to improve model performance, including:
     - Data augmentation (rotation, scaling, flipping)
     - Dropout layers to reduce overfitting
     - Increasing convolutional layers and filter sizes

### 2. **Transfer Learning (VGG16)**
   - Pre-trained **VGG16 model** was fine-tuned for jellyfish classification.
   - Different layers were frozen/unfrozen for comparison.
   - Fully connected layers were added for classification.

## Results
- **CNN models** struggled with **overfitting**, with a best validation accuracy of **0.67**.
- **Transfer Learning (VGG16)** achieved **higher validation accuracy (0.83)** and better generalization.
- **Data augmentation and dropout layers** helped improve CNN performance but couldn't outperform VGG16.

## Key Findings
- **Transfer learning is more effective for small datasets** compared to training a CNN from scratch.
- **Data augmentation and dropout** can help mitigate overfitting but may not fully solve it in CNN models.
- **Fine-tuning pre-trained models** allows better feature extraction, leading to improved classification accuracy.

## Authors
- **Ang Xu**
- **Jiazhen Jing**
- **Dinesha Dissanayake**

## Citation
If you use this work, please cite our report:

> Xu, A., Jing, J., & Dissanayake, D. *Comparison of Two Supervised Learning Methods for the Classification of Jellyfish Images*. (2024).

---

🔹 **Feel free to explore, contribute, and provide feedback!** 🚀
