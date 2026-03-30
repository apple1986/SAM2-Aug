# SAM2-Aug

**SAM2-Aug: Prior Knowledge-Based Augmentation for Target Volume Auto-Segmentation in Adaptive Radiation Therapy Using Segment Anything Model 2**

📄 Paper: https://arxiv.org/abs/2507.19282  
📦 Code: https://github.com/apple1986/SAM2-Aug

---

## 🚀 Overview

Accurate tumor segmentation is critical for adaptive radiation therapy (ART), yet it remains a time-consuming and labor-intensive task with significant inter-observer variability. 

Recent foundation models such as **Segment Anything Model 2 (SAM2)** have demonstrated strong capabilities in prompt-based segmentation. However, their performance in tumor segmentation tasks—especially in clinical ART workflows—still faces challenges in accuracy and robustness.

**SAM2-Aug** addresses these limitations by introducing **prior knowledge-based augmentation strategies** to enhance SAM2 for medical image segmentation.

---

## 🧠 Key Contributions

- ✅ **Prior Knowledge Integration**  
  Leverages prior MR images and corresponding annotations to provide contextual guidance for segmentation.

- ✅ **Prompt Augmentation Strategy**  
  Improves robustness through:
  - Random bounding box expansion  
  - Mask erosion and dilation during fine-tuning  

- ✅ **Strong Generalization**  
  Demonstrates cross-dataset and cross-modality robustness without large-scale retraining.

---

## 🧪 Datasets

The model was trained and evaluated on the following datasets:

- **One-Seq-Liver**
  - 115 MRI scans  
  - 31 liver cancer patients  
  - Single imaging sequence  

- **Mix-Seq-Abdomen**
  - 88 MRI scans  
  - 28 patients with abdominal tumors  
  - Mixed imaging sequences  

- **Mix-Seq-Brain**
  - 86 MRI scans  
  - 37 patients with brain tumors  
  - Mixed imaging sequences  


---

## 📊 Results

SAM2-Aug outperforms state-of-the-art convolutional, transformer-based, and prompt-driven segmentation models across all datasets:

| Task              | Dice Score (mean ± std) |
|-------------------|------------------------|
| ITV (Liver)       | 0.86 ± 0.08           |
| ITV (Abdomen)     | 0.89 ± 0.06           |
| CTV (Brain)       | 0.90 ± 0.07           |

### Highlights
- Robust to complex and variable tumor boundaries  
- Strong cross-modality generalization  
- No need for large-scale retraining  
- Improved segmentation quality in challenging ART scenarios  

---

## 🏥 Conclusion

By integrating prior imaging information and enhancing prompt diversity, **SAM2-Aug** provides:

- ✔️ Accurate tumor segmentation  
- ✔️ Improved robustness  
- ✔️ Strong generalization  

These advancements offer a promising path toward **more efficient and precise adaptive radiation therapy workflows**.

---

## 🔧 Code & Models

The implementation and trained models are available at:

👉 https://github.com/apple1986/SAM2-Aug

---

## 🙏 Acknowledgements

This work builds upon:

- **Segment Anything Model 2 (SAM2)** by Meta AI  

---

## 📚 Citation

If you find this work useful, please cite:
Xu, Guoping, et al.
"SAM2-Aug: Prior knowledge-based Augmentation for Target Volume Auto-Segmentation
in Adaptive Radiation Therapy Using Segment Anything Model 2."
arXiv preprint arXiv:2507.19282 (2025).

## ⭐️ Star the Repo

If you find this project helpful, consider giving it a ⭐️ on GitHub!
