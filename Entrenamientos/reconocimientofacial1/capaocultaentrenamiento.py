import cv2 as cv
import os
import numpy as np
from time import time

# Path to the directory containing training data
data_path = '/home/monti/Escritorio/InHome/PythonImageProcess/Entrenamientos/reconocimientofacial1/Data'

# List all directories (each representing a person) in the data directory
data_list = os.listdir(data_path)
print('Data:', data_list)

# Initialize lists to store face images and corresponding IDs
ids = []
face_data = []

# Record start time for data reading
start_time = time()

# Loop through each directory (person) in the data directory
for person_id, folder_name in enumerate(data_list):
    # Complete path to the person's directory
    complete_path = os.path.join(data_path, folder_name)
    print('Starting reading...')
    
    # Loop through each image file in the person's directory
    for file_name in os.listdir(complete_path):
        print('Images:', folder_name + '/' + file_name)
        
        # Append the person's ID to the IDs list
        ids.append(person_id)
        
        # Read the image in grayscale
        face_image = cv.imread(os.path.join(complete_path, file_name), 0)
        
        # Append the face image to the face data list
        face_data.append(face_image)

# Calculate the total time taken for data reading
total_reading_time = time() - start_time
print('Total reading time:', total_reading_time)

# Create an instance of the Face Recognizer object and choose the recognition algorithm
eigen_face_recognizer = cv.face.EigenFaceRecognizer_create()
# eigen_face_recognizer = cv.face.FisherFaceRecognizer_create()
# eigen_face_recognizer = cv.face.LBPHFaceRecognizer_create()

print('Starting training... Please wait.')
# Train the recognizer using the face data and corresponding IDs
eigen_face_recognizer.train(face_data, np.array(ids))

# Calculate the total time taken for training
total_training_time = time() - start_time
print('Total training time:', total_training_time)

# Save the trained model to a file
eigen_face_recognizer.write('EigenFaceRecognizerTraining.xml')
print('Training completed.')
