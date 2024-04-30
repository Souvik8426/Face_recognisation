import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)

nose_cascade_path = './haarcascade_mcs_nose.xml'
mouth_cascade_path = './haarcascade_mcs_mouth.xml'

while True:
    ret, frame = video_capture.read()

    # Convert the frame from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use face_recognition library to detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw rectangles around the detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Label the face
        cv2.putText(frame, 'Face', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract the face region from the frame
        face_image = frame[top:bottom, left:right]

        # Use Haar cascade classifiers for eyes, nose, and mouth detection within the face region
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
        nose = cv2.CascadeClassifier(nose_cascade_path).detectMultiScale(gray_face, scaleFactor=1.3, minNeighbors=5)
        mouth = cv2.CascadeClassifier(mouth_cascade_path).detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)

        for (ex, ey, ew, eh) in eyes:
            confidence = (1 - (ew * eh) / (face_image.shape[0] * face_image.shape[1])) * 100
            if confidence > 20:  # Confidence threshold
                cv2.rectangle(face_image, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                cv2.putText(frame, 'Eye', (left + ex, top + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(face_image, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
            cv2.putText(frame, 'Nose', (left + nx, top + ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for (sx, sy, sw, sh) in mouth:
            confidence = (1 - (sw * sh) / (face_image.shape[0] * face_image.shape[1])) * 100
            if confidence > 15 and sy > face_image.shape[0] / 2:  # Improved confidence threshold and position check
                cv2.rectangle(face_image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                cv2.putText(frame, 'Mouth', (left + sx, top + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Facial Features Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
