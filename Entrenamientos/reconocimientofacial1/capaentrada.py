import cv2 as cv
import os
import imutils

# Model name and path to the directory containing training data
model_name = 'FotosIsma'
data_path = '/home/monti/Escritorio/InHome/PythonImageProcess/Entrenamientos/reconocimientofacial1/Data'
complete_path = os.path.join(data_path, model_name)

# Create the directory if it doesn't exist
if not os.path.exists(complete_path):
    os.makedirs(complete_path)

# Open the camera for capturing video
camera = cv.VideoCapture("videoauron.mp4")

# Load the Haar cascade classifier for face detection
face_cascade = cv.CascadeClassifier('/home/monti/Escritorio/InHome/PythonImageProcess/Entrenamientos/entrenamientos_opencv_ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

# Initialize the ID counter
id_counter = 0

# Start capturing frames from the camera
while True:
    # Read a frame from the camera
    response, frame = camera.read()
    
    # If there are no more frames, break out of the loop
    if not response:
        break
    
    # Resize the frame to a width of 640 pixels
    frame = imutils.resize(frame, width=640)
    
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) containing the face
        face_roi = frame_copy[y:y + h, x:x + w]
        
        # Resize the ROI to 160x160 pixels
        resized_face_roi = cv.resize(face_roi, (160, 160), interpolation=cv.INTER_CUBIC)
        
        # Save the resized face ROI as an image file
        file_path = os.path.join(complete_path, f'image_{id_counter}.jpg')
        cv.imwrite(file_path, resized_face_roi)
        
        # Increment the ID counter
        id_counter += 1

        # Show a visual message of registration
        cv.putText(frame, 'Face registered', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    # Display the frame with annotations
    cv.imshow("Face Result", frame)

    # Check for 's' key press to stop capturing
    if id_counter == 351 or cv.waitKey(1) == ord('s'):
        break

# Release the camera object
camera.release()

# Close all OpenCV windows
cv.destroyAllWindows()
