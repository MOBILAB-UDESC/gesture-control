import cv2
import time
import os.path
from libs.face_detector import FaceDetector

import face_recognition
import pickle

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
print("[INFO] detecting faces...")

detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=0.5,
                            overlap_thr=0.7)

data = pickle.loads(open("encodings/data.db", "rb").read())

try:
    while True:
        start_time = time.time()
        conectado, image = camera.read()

        boxes, scores = detector.inference(image)
        # detector.draw_bboxes(image, boxes)

        encodings = face_recognition.face_encodings(image, boxes)

         # # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((x_min, y_min, x_max, y_max), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            y = y_min - 15 if y_min - 15 > 15 else y_min + 15
            cv2.putText(image, name, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        
        cv2.putText(image, f"FPS: {1 / (time.time() - start_time):.2f}", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

        # show the output image
        cv2.imshow("Video", image)
        if cv2.waitKey(1) == ord('q'): break

finally:
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas pela OpenCV