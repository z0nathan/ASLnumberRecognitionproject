#데이터 증강
import numpy as np
import os

def rotate(landmarks_,angle):#angle: radian
    rotation_matrix_z = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
    ])
    for i in range(len(landmarks_)):
        landmarks_[i] = np.dot(rotation_matrix_z, landmarks_[i])

def mirror(landmarks_):
    for i in range(len(landmarks_)):
        landmarks_[i] = landmarks_[i] * np.array([-1, 1, 1])

# Define the dataset path (where all the .npy files are stored)
dataset_dir = './termProjectData/'  # Root folder containing gesture data

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
        n=len(os.listdir(label_dir))
        for file in os.listdir(label_dir):
            if file.endswith(".npy"):
                file_path = os.path.join(label_dir, file)
                landmarks = np.load(file_path)  # Load the landmarks data from the .npy file
                
                # Check if the data is valid and does not contain outliers
                if is_valid_landmark(landmarks):
                    for i in range(len(landmarks)):
                        landmarks[i]=landmarks[i]-landmarks[0]

                    #data expand
                    expandedLandmarks=np.copy(landmarks)
                    for j in range(3):
                        rotate(expandedLandmarks,0.275)
                        newfilepath='./termProjectData/'+gesture_label+'/'+str(n)+'.npy'
                        n+=1
                        np.save(newfilepath, expandedLandmarks)
                        mirror(expandedLandmarks)
                        newfilepath='./termProjectData/'+gesture_label+'/'+str(n)+'.npy'
                        n+=1
                        np.save(newfilepath, expandedLandmarks)

                        rotate(expandedLandmarks,-0.275)
                        newfilepath='./termProjectData/'+gesture_label+'/'+str(n)+'.npy'
                        n+=1
                        np.save(newfilepath, expandedLandmarks)
                        mirror(expandedLandmarks)
                        newfilepath='./termProjectData/'+gesture_label+'/'+str(n)+'.npy'
                        n+=1
                        np.save(newfilepath, expandedLandmarks)
                else:
                    print(f"Outlier detected in file: {file_path}, skipping...")
            print("\n")
            
print("data expansion done..")




