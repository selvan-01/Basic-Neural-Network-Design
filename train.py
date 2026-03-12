"""
PIMA Indians Diabetes Dataset Features

1. Number of times pregnant
2. Plasma glucose concentration (2 hours in OGTT)
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (BMI)
7. Diabetes pedigree function
8. Age (years)

Target:
9. Class variable (0 = Non-Diabetic, 1 = Diabetic)
"""

# ===============================
# Import Required Libraries
# ===============================
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# ===============================
# Load Dataset
# ===============================

# Load the CSV dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split dataset into input (X) and output (Y)
x = dataset[:, 0:8]   # First 8 columns → Input features
y = dataset[:, 8]     # Last column → Target output

print("Sample Input Data:")
print(x)


# ===============================
# Build Neural Network Model
# ===============================

# Initialize Sequential Model
model = Sequential()

# Hidden Layer 1
model.add(Dense(
    12,                # Number of neurons
    input_dim=8,       # Number of input features
    activation='relu'  # Activation function
))

# Hidden Layer 2
model.add(Dense(
    8,
    activation='relu'
))

# Output Layer
model.add(Dense(
    1,
    activation='sigmoid'   # Used for binary classification
))


# ===============================
# Compile Model
# ===============================

model.compile(
    loss='binary_crossentropy',   # Loss function for binary classification
    optimizer='adam',             # Optimization algorithm
    metrics=['accuracy']          # Evaluation metric
)


# ===============================
# Train Model
# ===============================

print("\nTraining Model...\n")

model.fit(
    x,
    y,
    epochs=40,        # Number of training iterations
    batch_size=10     # Number of samples per gradient update
)


# ===============================
# Evaluate Model
# ===============================

loss, accuracy = model.evaluate(x, y)

print("\nModel Accuracy: %.2f%%" % (accuracy * 100))


# ===============================
# Save Model
# ===============================

# Save model architecture
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("model.h5")

print("\nModel saved successfully to disk")