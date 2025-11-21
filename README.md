# Matrix & Tensor Decomposition for Video Background–Foreground Separation

## 📌 Project Overview
This project compares **matrix-based** and **tensor-based** decomposition methods for separating **background** and **foreground** in grayscale videos. The work evaluates six algorithms using both **synthetic data** and **real video frames**, measuring reconstruction accuracy, runtime efficiency, and visual separation quality.

This implementation is based on the research documented in *“Comparison of Matrix and Tensor Decomposition in Separation of Video Background and Foreground”* by Felix Sjöholm & Yug Bhavsar.

---

## 🎯 Objective
To determine whether videos are better processed as:
- **Matrices** (vectorized frames), or  
- **Tensors** (preserving spatial-temporal structure)

for improved:
- Background reconstruction  
- Foreground extraction  
- Noise removal  
- Computational efficiency  

---

## 📂 Methods Implemented

### **Matrix-Based Methods**
#### 1. RPCA (Robust PCA)
- Uses Principal Component Pursuit  
- Recovers low-rank background + sparse foreground  
- Best accuracy across methods  

#### 2. CUR Decomposition
- Samples key rows/columns  
- Very fast but less accurate  

#### 3. IRCUR (Iterated Robust CUR)
- CUR + iterative thresholding  
- More robust than CUR  
- Sensitive to implementation and noise  

---

### **Tensor-Based Methods**
#### 1. CP Decomposition (ALS)
- Represents tensor as sum of rank-1 tensors  
- Efficient but not ideal for video structure  

#### 2. Tucker Decomposition (HOOI)
- Tensor generalization of PCA  
- Strong reconstruction accuracy  

#### 3. RTCUR (Tensor Fiber CUR)
- Extends CUR to tensors via fiber sampling  
- Lowest accuracy in experiments  

---

## 🧪 Datasets Used

### **Synthetic Data**
- 10 grayscale frames (128×128)  
- Gaussian noise added  
- Sparse high-magnitude outliers  

### **Real Video**
- Grayscale MP4  
- First 20 frames used  
- Frames downsampled for computation  

---

## 📊 Results Summary

### **Matrix Methods — Synthetic Data**

| Method | SNR (dB) |
|--------|-----------|
| **RPCA** | **35.89** |
| CUR | 31.44 |
| IRCUR | 31.44 |

**RPCA** produces the cleanest background and strongest foreground isolation.

---

### **Tensor Methods — Synthetic Data**

| Method | SNR (dB) |
|--------|-----------|
| **Tucker** | **31.59** |
| CP | 31.46 |
| RTCUR | 18.22 |

**Tucker** outperforms other tensor decompositions.

---

### **Real Video Background–Foreground Separation**
- **RPCA**: Best reconstruction + clearest motion segmentation  
- **CUR / IRCUR**: Detect motion but contain sampling artifacts  
- **Tucker / CP**: Preserve background but weak foreground separation  
- **RTCUR**: Heavy distortions  

---

## ⚙️ Technologies Used
- Python 3  
- NumPy  
- SciPy  
- TensorLy  
- OpenCV  
- Matplotlib  
- Scikit-Image  

---
## 2. Run the script
python matrix_tensor_video_decomposition.py

## 3. Outputs generated
- Background reconstructions
- Foreground maps
- Error heatmaps
- SNR/SSIM metrics
- Comparison bar charts & radar plots

## 🧠 Key Findings
- RPCA is the most reliable and accurate for video background subtraction.
- Tucker is the strongest tensor-based method.
- CUR/IRCUR offer speed over accuracy.
- RTCUR performs poorly with small datasets and high noise.
- Tensor models show promise but require larger datasets for full effectiveness.

## 📝 References
All references are included in the research report.

## 👥 Authors
- Felix Sjöholm
- Yug Bhavsar
