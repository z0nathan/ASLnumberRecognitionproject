#비디오서 제스쳐 인식 & 이미지 변동별 출력 
import cv2
import numpy as np
import tensorflow as tf
from mediapipe import solutions as mp_solutions

import imagehash
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('__gesture_recognition_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp_solutions.hands
mp_drawing = mp_solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define gesture labels (make sure they match your training labels)
gesture_labels = ['1', '2', '3', '4', '5', 
                  '6', '7', '8', '9', '10']

# Open the MP4 video file
video_path = '/home/kmj/Documents/2024Univ/2024AIsysDedign/termProject/asl_numbers_final.mp4'#'/home/kmj/Documents/2024Univ/2024AIsysDedign/termProject/Testset_ASL_numbers.mp4' 
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Initialize variables for frame-to-frame comparison
previous_frame = None
frame_count = 0
img_difference=0
prev_difference=10
change_count = 0  # Counter for significant changes
recogTerm=0     # for the term to recognize gesture
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Convert the current frame to grayscale for pixel comparison
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compare with the previous frame
    if previous_frame is not None:
        h,w,_=frame.shape
        c1 = abs(np.sum(cv2.mean(frame[0:h//2,0:w//2]))-np.sum(cv2.mean(previous_frame[0:h//2,0:w//2])))
        c2 = abs(np.sum(cv2.mean(frame[0:h//2,w//2:w]))-np.sum(cv2.mean(previous_frame[0:h//2,w//2:w])))
        c3 = abs(np.sum(cv2.mean(frame[h//2:h,0:w//2]))-np.sum(cv2.mean(previous_frame[h//2:h,0:w//2])))
        c4 = abs(np.sum(cv2.mean(frame[h//2:h,w//2:w]))-np.sum(cv2.mean(previous_frame[h//2:h,w//2:w])))
        prev_difference=img_difference
        img_difference = c1+c2+c3+c4
        #img_difference = np.sum(cv2.absdiff(gray_frame, previous_frame)[100:200])
        # Sum up pixel differences
        #img_difference = np.sum(frame_difference)

    # Update the previous frame for the next iteration
    previous_frame = frame

    # Process the frame with MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the current frame
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Ensure the landmarks have the correct shape
            if landmarks.shape == (21, 3):
                # Normalize the landmarks
                for i in range(len(landmarks)):
                    landmarks[i] = landmarks[i] - landmarks[0]
                landmarks = landmarks / (np.linalg.norm(landmarks[1], axis=0) + 0.0001)

                # Add batch dimension and pass to the model
                input_data = np.expand_dims(landmarks, axis=0)
                prediction = model.predict(input_data, verbose=0)
                gesture_id = np.argmax(prediction)
                gesture_name = gesture_labels[gesture_id]

                # Display the predicted gesture on the frame
                cv2.putText(frame, f'{change_count}: {gesture_name}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                if recogTerm==1:            
                    change_count += 1
                    #print(f"{gesture_name}, ")
                    print(f"{change_count}:  {gesture_name}")
                if recogTerm>=1:
                    recogTerm-=1
                elif abs(prev_difference-img_difference)>3.23:
                    recogTerm=2


            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Pixel-Based Frame Change Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
