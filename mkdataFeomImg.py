#이미지서 랜드마크 ndarray 추출
import numpy as np
import mediapipe as mp
import cv2
import os

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Directory containing images for gesture data
gesture_label = 7  # Set the gesture label for the images
image_folder = './termProject/gesture_images/'+str(gesture_label)+'/'  # Path to your folder with images
output_folder = './termProjectData/'  # Path to save .npy files

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to save landmarks to a .npy file
def save_landmarks(landmarks, label, filename):
    # Define the directory for the label
    label_folder = os.path.join(output_folder, f'{label}_gesture_data')
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    # Save the landmarks with the corresponding filename
    np.save(os.path.join(label_folder, filename), landmarks)
    print(f"Landmarks for label '{label}' saved as {filename}.npy")

# Process each image in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
n = 0  # Counter for saved files

for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # Convert the image to RGB (MediaPipe expects RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to get hand landmarks
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the landmarks as a list of (x, y, z) coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

            # Convert landmarks to a NumPy array
            landmarks = np.array(landmarks)

            # Save the landmarks
            save_landmarks(landmarks, gesture_label, f'{n}')
            n += 1

    else:
        print
