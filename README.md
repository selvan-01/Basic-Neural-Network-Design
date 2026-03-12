# Diabetes Prediction using Neural Network

This project implements a **Binary Classification Neural Network** using **Keras and TensorFlow** to predict whether a patient has diabetes based on medical attributes.

---

## Dataset

The dataset used is the **PIMA Indians Diabetes Dataset**.

Features:

1. Number of times pregnant
2. Plasma glucose concentration
3. Diastolic blood pressure
4. Triceps skin fold thickness
5. Serum insulin
6. Body Mass Index (BMI)
7. Diabetes pedigree function
8. Age

Target:
- 0 → Non-Diabetic
- 1 → Diabetic

---

## Technologies Used

- Python
- NumPy
- TensorFlow
- Keras
- Scikit-learn

---

## Neural Network Architecture

Input Layer:
8 Features

Hidden Layer 1:
12 neurons (ReLU activation)

Hidden Layer 2:
8 neurons (ReLU activation)

Output Layer:
1 neuron (Sigmoid activation)

Loss Function:
Binary Crossentropy

Optimizer:
Adam

---

## Project Files

train.py  
Used to train the neural network model.

test.py  
Loads the trained model and makes predictions.

pima-indians-diabetes.csv  
Dataset used for training.

model.json  
Saved model architecture.

model.h5  
Saved model weights.

requirements.txt  
Python dependencies.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Diabetes-Prediction-Neural-Network.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python train.py
```

---

## Run Prediction

```bash
python test.py
```

---

## Example Output

```
Model Accuracy: 78.50%
[6.0,148.0,72.0,35.0,0.0,33.6,0.627,50.0] => Predicted: 1 | Expected: 1
```

---

## Future Improvements

- Add Train/Test split
- Data normalization
- Confusion matrix
- Accuracy visualization
- Deploy using Flask or Streamlit

---

## Author

Senthamil Selvan  
Computer Science Engineer  
AI & Data Analytics Enthusiast
