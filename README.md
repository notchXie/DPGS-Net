# Dual Prior-Guided Two-Stage Framework for Small-Sample Ultrasound Image Segmentation

> Official PyTorch implementation of our method proposed in the paper:  
> **"Dual Prior-Guided Two-Stage Framework for Small-Sample Ultrasound Image Segmentation"**  
> 📌 Accepted/Submitted to MICCAI 2025  
> 📅 Project based on the DDTI dataset for medical ultrasound image segmentation.

## 🧠 Overview

Ultrasound image segmentation is crucial for medical-assisted diagnosis but faces challenges like:

- High noise and image artifacts  
- Large morphological variation across views  
- Feature contradiction due to different acquisition angles  
- Difficulty generalizing under small-sample data conditions

To tackle these, we propose a **Dual Prior-Guided Two-Stage Segmentation Framework**, which enhances segmentation robustness and generalization under limited data.  

### 🔧 Key Contributions

#### Stage 1: Domain Adaptation Pretraining with Prior Guidance

- Uses **prior classification on small-sample data** to guide **domain adaptation pretraining** on large-scale datasets of the same category.
- Introduces **dynamic class balancing** to mitigate data distribution bias during adaptation.

#### Stage 2: Multi-Level Feature Fusion Segmentation

- ⚙️ **Multi-branch Convolutional Parallel Attention (MCPA):**  
  Captures multi-scale contextual features using parallel dilated convolutions and dual (channel + spatial) attention.

- 🌀 **Multi-scale Fusion Dilated Convolution (MFDC):**  
  Enhances boundary representation through hierarchical receptive fields and dilated convolutions.

- 🧩 **Enhanced Feature Decoding (EFD):**  
  Utilizes shallow high-resolution features via cross-layer compensation to restore spatial detail in deep features.

#### 🔁 Dual-Stream Interactive Architecture

- Bridges the classification and segmentation tasks.
- Features are exchanged via **cross-task attention**, improving semantic consistency and robustness.

