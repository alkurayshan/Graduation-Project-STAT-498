
<p align="center">
  <img src="KSU_MasterLogo_Colour_RGB.png" alt="King Saud University Logo" width="380" />
</p>

# Comparing Arabic BERT vs Logistic Regression — Graduation Project

This Streamlit application compares **Large Language Models (Arabic BERT)** with **Traditional Statistical Models (TF–IDF + Logistic Regression)** for **sentiment classification of Saudi banks’ tweets**.  

Both models are trained & evaluated using a **shared stratified split (70% train / 10% validation / 20% test)** to ensure a **fair and statistically valid comparison**.

---

##  Project Overview

The project includes:

- 🔹 Shared train/validation/test split for both models  
- 🔹 ROC–AUC, Confusion Matrix, Calibration (ECE), Brier Score  
- 🔹 McNemar test for statistical significance  
- 🔹 Interactive prediction using text & CSV files  
- 🔹 Full model introspection (**W**, **b**, logits, softmax) for both BERT & Logistic Regression  

---

##  Institutional Information

**King Saud University**  
**College of Science**  
**Department of Statistics & Operations Research**

This work is submitted as a **graduation requirement** for the  
**Bachelor’s Degree in Statistics**.

---

##  Student Information

| Role | Details |
|------|---------|
| **Student** | **Abdulrahman Mohammed Al-Kurayshan** |
| 📧 **University Email** | `443102096@student.ksu.edu.sa` |
| 📧 **Personal Email** | `alkurayshanabdulrahman@outlook.com` |
| 🔗 **LinkedIn** | https://www.linkedin.com/in/akurayshan/ |

---

##  Academic Supervisor

| Role | Details |
|------|---------|
| **Project Supervisor** | **Dr. Aaid Algahtani** |
| 📧 **Email** | `aalgahtani@ksu.edu.sa` |

---

##  Tech Stack

| Category | Tools / Libraries |
|----------|------------------|
| **Frontend** | Streamlit |
| **NLP Model** | Arabic BERT (`asafaya/bert-base-arabic`) |
| **Traditional Model** | TF-IDF + Logistic Regression |
| **Metrics** | ROC-AUC, Macro-F1, Calibration (ECE), Brier Score, McNemar Test |
| **Language** | Python (3.9+) |
| **Frameworks** | HuggingFace Transformers, Scikit-learn, PyTorch |

---
