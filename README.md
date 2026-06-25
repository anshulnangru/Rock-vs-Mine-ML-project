# 🌊 Rock vs Mine Prediction using Machine Learning

A Machine Learning classification project that predicts whether an object detected by SONAR signals is a **Rock** or a **Mine**.

The project uses sonar signal data and a trained machine learning model to classify underwater objects, making it useful for applications in marine navigation, defense systems, and underwater exploration.

---

## 📌 Project Overview

SONAR (Sound Navigation and Ranging) technology is widely used to detect objects underwater by sending sound waves and analyzing their reflections.

In this project, a Machine Learning model is trained on SONAR signal data to determine whether the detected object is:

* **Rock (R)**
* **Mine (M)**

The model learns patterns from historical sonar readings and predicts the class of unseen observations. The dataset contains 208 samples with 60 numerical sonar signal features and one target label.

---

## 🚀 Features

* Rock vs Mine Classification
* SONAR Signal Analysis
* Data Preprocessing
* Train-Test Data Splitting
* Machine Learning Model Training
* Model Evaluation
* Real-Time Prediction
* Binary Classification Problem

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Scikit-Learn

### Machine Learning Concepts

* Classification
* Data Preprocessing
* Feature Analysis
* Model Evaluation

---

## 📂 Dataset Information

The SONAR dataset contains:

* 208 observations
* 60 numerical features
* 1 target column

Target Classes:

| Label | Meaning |
| ----- | ------- |
| R     | Rock    |
| M     | Mine    |

Each feature represents the energy reflected from a particular sonar frequency band.

---

## ⚙️ Project Workflow

### 1. Data Collection

Load the SONAR dataset into a Pandas DataFrame.

```python
data = pd.read_csv('sonar_data.csv', header=None)
```

### 2. Data Preprocessing

* Check dataset shape
* Analyze feature distributions
* Separate features and labels

```python
X = data.drop(columns=60, axis=1)
Y = data[60]
```

### 3. Train-Test Split

Split the dataset into training and testing sets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,
    stratify=Y,
    random_state=1
)
```

### 4. Model Training

Train a Machine Learning classification model.

```python
model.fit(X_train, Y_train)
```

### 5. Model Evaluation

Evaluate model performance on training and testing data.

```python
accuracy_score(Y_test, prediction)
```

### 6. Prediction

Predict whether a new SONAR signal corresponds to a Rock or Mine.

```python
prediction = model.predict(input_data)
```

---

## 🔄 Machine Learning Pipeline

```text
SONAR Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Selection
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Prediction
      │
      ▼
Rock / Mine Classification
```

---

## 💻 Example Prediction

### Input

```text
[0.0200, 0.0371, 0.0428, ...]
```

### Output

```text
Mine Detected
```

or

```text
Rock Detected
```

---

## 📊 Model Evaluation

Common evaluation metrics used:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example:

```text
Training Accuracy : 83%
Testing Accuracy  : 76%
```

*(Actual accuracy may vary depending on train-test split and model used.)*

---

## 🌍 Real-World Applications

* Naval Defense Systems
* Underwater Mine Detection
* Marine Exploration
* Oceanographic Research
* Autonomous Underwater Vehicles (AUVs)

---

## 🎯 Learning Outcomes

Through this project I learned:

* Classification Algorithms
* Data Preprocessing
* Feature Analysis
* Model Evaluation
* SONAR Signal Data Handling
* End-to-End Machine Learning Workflow

---

## 📁 Project Structure

```text
Rock-vs-Mine-ML-project/
│
├── Rock_vs_Mine_Prediction.ipynb
├── sonar_data.csv
├── README.md
└── requirements.txt
```

---

## 🔮 Future Improvements

* Streamlit Web Application
* Deep Learning Models
* Hyperparameter Tuning
* Real-Time SONAR Analysis
* Model Comparison Dashboard
* Cloud Deployment

---

## 👨‍💻 Author

**Anshul Nangru**

BCA Student | Machine Learning Enthusiast

GitHub: https://github.com/anshulnangru

---

## ⭐ Support

If you found this project useful, consider giving the repository a star ⭐.
