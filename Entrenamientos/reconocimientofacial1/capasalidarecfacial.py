import cv2 as cv
import os
import imutils

# Path to the directory containing training data
data_path = '/home/monti/Escritorio/InHome/PythonImageProcess/Entrenamientos/reconocimientofacial1/Data'

# List all files in the data directory
data_list = os.listdir(data_path)

# Create an instance of the EigenFaceRecognizer
face_recognizer = cv.face.EigenFaceRecognizer_create()

# Load the trained EigenFaceRecognizer from the XML file
face_recognizer.read('EntrenamientoEigenFaceRecognizer.xml')

# Load the Haar cascade classifier for face detection
face_cascade = cv.CascadeClassifier('/home/monti/Escritorio/InHome/PythonImageProcess/Entrenamientos/entrenamientos_opencv_ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

# Open the video file for processing
video_capture = cv.VideoCapture("ElonMusk.mp4")

# Start processing frames from the video
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    
    # If there are no more frames, break out of the loop
    if not ret:
        break
    
    # Resize the frame to a width of 640 pixels
    frame = imutils.resize(frame, width=850)
    
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Create a copy of the grayscale frame
    gray_copy = gray.copy()
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray_copy[y:y+h, x:x+w]
        
        # Resize the ROI to 160x160 pixels
        resized_face_roi = cv.resize(face_roi, (160, 160), interpolation=cv.INTER_CUBIC)
        
        # Perform face recognition on the resized face ROI
        label, confidence = face_recognizer.predict(resized_face_roi)
        print(f'Prediction: Label={label}, Confidence={confidence:.2f}')  # Debug message
        
        # Draw text on the frame with the prediction result
        text = f'{data_list[label]} - Confidence: {confidence:.2f}'
        cv.putText(frame, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv.LINE_AA)

        # If the confidence level is below a certain threshold, display the recognized person's name
        if confidence < 9000:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv.putText(frame, "Not recognized", (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the frame with annotations
    cv.imshow("Results", frame)
    
    # Check for the 's' key press to stop processing
    if cv.waitKey(1) == ord('s'):
        break

# Release the video capture object
video_capture.release()

# Close all OpenCV windows
cv.destroyAllWindows()
