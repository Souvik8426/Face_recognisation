import cv2
import face_recognition

image_paths = ['./Assets/sample1.jpg','./Assets/sample1.jpeg','./Assets/sample2.jpeg','./Assets/sample3.jpg','./Assets/sample4.jpg']

for image_path in image_paths:
    image = cv2.imread(image_path)

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use face_recognition library to detect faces in the image
    face_locations = face_recognition.face_locations(rgb_image)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Extract the face region from the image
        face_image = image[top:bottom, left:right]

        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(gray_face)
        smiles = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml').detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)

        # Draw rectangles around the detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_image, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Draw rectangles around the detected smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow('Facial Features Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

