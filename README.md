# Matrix & Tensor Video Decomposition for Background Separation

This repository implements six advanced decomposition algorithms for separating **background (low-rank)** and **foreground (sparse/dynamic)** components of video data using both **matrix-based** and **tensor-based** methods.

The project includes:
- Synthetic experiments  
- Tensor synthetic experiments  
- Real video experiments  
- SNR, SSIM, runtime comparisons  
- Heatmaps, bar plots, and radar charts  
- A complete Colab notebook for reproducibility  

---

## 📌 Methods Implemented

### **Matrix-Based Methods**
- **RPCA** – Robust PCA using Principal Component Pursuit (ALM)
- **CUR Decomposition** – Column-Row low-rank approximation
- **IRCUR** – Iterated Robust CUR using energy-based row/column selection

### **Tensor-Based Methods**
- **CP Decomposition** – CANDECOMP/PARAFAC (ALS-based)
- **Tucker Decomposition** – HOSVD / HOOI
- **RTCUR** – Tensor True Fiber CUR decomposition
