#데이터 학습 
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Define the dataset path (where all the .npy files are stored)
dataset_dir = './termProjectData'  # Root folder containing gesture data

# Initialize lists for data and labels
data = []
labels = []

# Define expected landmarks count
expected_landmarks_count = 63  # 21 landmarks with 3 coordinates (x, y, z)

# Function to check if the landmarks are valid
def is_valid_landmark(landmarks):
    # Check if the array has the expected shape (21, 3)
    return landmarks.shape == (21, 3)




# Loop over each folder corresponding to a gesture label (1 to 10)
for gesture_label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, gesture_label)
    
    # Check if it's a folder
    if os.path.isdir(label_dir):
        for file in os.listdir(label_dir):
            if file.endswith(".npy"):
                file_path = os.path.join(label_dir, file)
                landmarks = np.load(file_path)  # Load the landmarks data from the .npy file
                
                # Check if the data is valid and does not contain outliers
                if is_valid_landmark(landmarks):
                    data.append(landmarks)
                    label = int(gesture_label.split('_')[0])  # Extract the number before '_gesture_data'
                    labels.append(label)
                else:
                    print(f"Outlier detected in file: {file_path}, skipping...")

print(f"Total valid data samples: {len(data)}")  # Print the number of valid data samples

# If there is no valid data, raise an error
if len(data) == 0:
    raise ValueError("No valid data found. Please check your .npy files.")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels (e.g., 0 -> 0, 1 -> 1, etc.)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),  # Flatten (21, 3) into (63,)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # Output layer
])


# Compile the model with loss function, optimizer, and metrics
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Calculate class weights to handle potential class imbalance
class_weights_array = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

# Convert the class weights array into a dictionary
class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}

# Train the model
model.fit(X_train, y_train, epochs=120, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weights)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Optionally, save the trained model for later use
model.save('gesture_recognition_model.h5')
