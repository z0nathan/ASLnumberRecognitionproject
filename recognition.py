import cv2
import numpy as np
import tensorflow as tf
from mediapipe import solutions as mp_solutions

def modifyInput(landmarks):
    landmarks_=landmarks
    for i in range(len(landmarks_)):
        landmarks_[i]=landmarks_[i]-landmarks_[0]
        landmarks_=landmarks_/np.linalg.norm(landmarks_[1],axis=0)#regularize by length between 0-1 landmark 
        landmarksWithCosMag_=np.empty((21,4))
        landmarksWithCosMag_[0]=np.concatenate([landmarks_[0],np.array([0])])
        landmarksWithCosMag_[1]=np.concatenate([landmarks_[1],np.array([0])])
        for i in range(2,len(landmarks_)):
            landmarksWithCosMag_[i]=np.concatenate([landmarks_[i], np.array([np.dot(landmarks_[i],landmarks_[1])/(np.linalg.norm(landmarks_[i]))])])
    return landmarksWithCosMag_


# Load the trained model
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define gesture labels (ensure they match your training labels)
gesture_labels = ['Gesture 1', 'Gesture 2', 'Gesture 3', 'Gesture 4', 'Gesture 5', 
                  'Gesture 6', 'Gesture 7', 'Gesture 8', 'Gesture 9', 'Gesture 10']

# Open a video file or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and preprocess the frame for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks and format as (21, 3)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)

            # Ensure the landmarks have the correct shape
            if landmarks.shape == (21, 3):
                for i in range(len(landmarks)):
                    landmarks[i]=landmarks[i]-landmarks[0]
                landmarks=landmarks/(np.linalg.norm(landmarks[1],axis=0)+0.0001)#regularize by length between 0-1 landmark
                # Add batch dimension and predict gesture
                input_data = np.expand_dims(landmarks, axis=0)
                prediction = model.predict(input_data)
                gesture_id = np.argmax(prediction)
                gesture_name = gesture_labels[gesture_id]

                # Draw the gesture name on the frame
                cv2.putText(frame, f'Gesture: {gesture_name}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw hand landmarks on the frame
            mp_solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the video frame with predictions
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()



