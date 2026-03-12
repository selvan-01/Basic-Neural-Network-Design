# ===============================
# Import Libraries
# ===============================
from numpy import loadtxt
from keras.models import model_from_json


# ===============================
# Load Dataset
# ===============================

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

x = dataset[:, 0:8]   # Input features
y = dataset[:, 8]     # Actual labels


# ===============================
# Load Saved Model
# ===============================

# Load model architecture
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Recreate model
model = model_from_json(loaded_model_json)

# Load weights
model.load_weights("model.h5")

print("Model loaded successfully from disk")


# ===============================
# Make Predictions
# ===============================

predictions = model.predict(x)


# ===============================
# Display Predictions
# ===============================

print("\nSample Predictions:\n")

for i in range(5, 10):

    # Convert probability to binary class
    predicted_class = int(predictions[i] > 0.5)

    print(
        "%s => Predicted: %d | Expected: %d"
        % (x[i].tolist(), predicted_class, y[i])
    )