#웹캠기반 데이터생성 
import numpy as np
import mediapipe as mp
import cv2
import os

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open webcam
cap = cv2.VideoCapture(0)

# Function to save landmarks and labels to a .npy file with the gesture label as the filename
def save_landmarks(landmarks, label, n):
    # Define the directory path
    dir_path = f'./termProjectData/{label}_gesture_data/'
    
    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Define the file path for saving the landmarks
    dataset_path = f'{dir_path}{n}.npy'

    # Save the landmarks with the corresponding label in a .npy file
    np.save(dataset_path, landmarks)
    print(f"Landmarks for label '{label}' saved to {dataset_path}")

l=6
n = len(os.listdir(f'./termProjectData/{l}_gesture_data/'))
# Collect landmarks and save when space is pressed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert the image to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks as a list of coordinates (x, y, z)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

                # Convert normalized coordinates to pixel coordinates
                h, w, c = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Draw a circle at each landmark
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circle

            # Draw the connections between the landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display the frame with landmarks
            cv2.imshow("Hand Landmark", frame)

            # Wait for key press to save landmarks data
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar key to save data
                label = l  # label for the gesture (you can change this as per your need)
                landmarks = np.array(landmarks)  # Convert landmarks to numpy array
                save_landmarks(landmarks, label, n)  # Save landmarks to a file with label as the filename
                n += 1
    # Exit the loop if 'q' is pressed
    # if key == ord('q'):
    #    break

# Release resources
cap.release()
cv2.destroyAllWindows()
