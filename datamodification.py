#데이터 전처리
import numpy as np
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
                    for i in range(len(landmarks)):
                        landmarks[i]=landmarks[i]-landmarks[0]
                    landmarks=landmarks/np.linalg.norm(landmarks[1],axis=0)#regularize by length between 0-1 landmark 
                    '''
                    landmarksWithCosMag=np.empty((21,4))
                    landmarksWithCosMag[0]=np.concatenate([landmarks[0],np.array([0])])
                    landmarksWithCosMag[1]=np.concatenate([landmarks[1],np.array([0])])
                    for i in range(2,len(landmarks)):
                        landmarksWithCosMag[i]=np.concatenate([landmarks[i], np.array([np.dot(landmarks[i],landmarks[1])/(np.linalg.norm(landmarks[i]))])])
                        #append cosine magnitude.. between vector 0-1, 0-i... magnitude of 0-1 is 1...  
                    #print(landmarksWithCosMag)
                    np.save(file_path, landmarksWithCosMag)
                    '''
                    np.save(file_path, landmarks)
                else:
                    print(f"Outlier detected in file: {file_path}, skipping...")
            print("\n")
            
print("data modification done..")




