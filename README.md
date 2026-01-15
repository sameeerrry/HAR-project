# 🚀 Human Activity Recognition (HAR) Web App

Smartphone-based recognition of human activities and postural transitions using **classical ML** and **LSTM** models.  
Built with Flask, scikit-learn, TensorFlow/Keras, and pandas.  
Predict human activities in real time via a clean web interface. 🔍📱

  ![image](https://github.com/user-attachments/assets/11c4ac75-6e02-446a-9fd9-33c521c28442)
  ![image](https://github.com/user-attachments/assets/dbdeb0c4-48df-438e-814e-b169e8c6c324)

---

## 📜 Table of Contents
- [✨ Features](#-features)
- [⚡ Quick Start](#-quick-start)
- [🕹️ Usage](#usage)
- [🏷️ Supported Activities](#supported-activities)
- [📁 Directory Structure](#-directory-structure)
- [🎯 Markdown Demo](#markdown-demo)
- [🧠 Credits](#-credits)


---

## ✨ Features

- **Multiple Models**:
  - Decision Tree
  - Linear SVC
  - Logistic Regression
  - RBF SVM
  - Random Forest
  - LSTM (TensorFlow/Keras)

- **Flexible Input**:
  - Single Sample (JSON)
  - Batch Input (CSV)

- **User-Friendly Interface**
- **Clear Error Handling** 🔥

---

## ⚡ Quick Start
[Dataset Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

```bash
git clone https://github.com/sameeerrry/HAR-project.git
cd your-har-project
pip install -r requirements.txt
python app.py
```

🚀 Open your browser at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🕹️ Usage

### 🔹 Single Sample Prediction

1. Paste a JSON array of features
   - 561 values for classical models
   - 1152 values for LSTM model
2. Select your model
3. Click `Predict`

### 🔹 Batch Prediction (CSV)

1. Upload a `.csv` file (no header, each row = 1 sample)
2. Select model
3. Click `Predict CSV`

> ⚠️ Use sample data from the **UCI HAR Dataset**

---

## 🏷️ Supported Activities

| Label | Activity |
|-------|----------|
| 0     | Walking |
| 1     | Walking Upstairs |
| 2     | Walking Downstairs |
| 3     | Sitting |
| 4     | Standing |
| 5     | Laying |

---

## 📁 Directory Structure

```
.
├── app.py
├── models/
│   ├── decision_tree_model.pkl
│   ├── linear_svc_model.pkl
│   ├── log_reg_model.pkl
│   ├── rbf_svm_model.pkl
│   ├── random_forest_model.pkl
│   └── lstml3_model.h5
├── templates/
│   └── index.html
├── utils/
│   └── preprocess.py
├── requirements.txt
└── README.md
```


---

## 🧠 Credits

- Based on the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Built with ❤️ using Flask, TensorFlow/Keras, scikit-learn, and pandas.

---
