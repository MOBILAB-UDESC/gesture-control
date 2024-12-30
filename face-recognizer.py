import cv2
import time
from libs.face_detector import FaceDetector

from deepface import DeepFace

# Function to crop an image based on a bounding box
def crop_image(image, bounding_box):
    height, width, _ = image.shape
    x_min, y_min, x_max, y_max = bounding_box

    new_x = x_min - (x_max - x_min) * 0.2
    new_y = y_min - (y_max - y_min) * 0.3
    new_w = x_max + (x_max - x_min) * 0.3
    new_h = y_max + (y_max - y_min) * 0.1

    cropped_image = image[max(int(new_y), 0):min(int(new_h), height - 1),
                          max(int(new_x), 0):min(int(new_w), width - 1)]
    
    return cropped_image

def recognize_faces(image, boxes):

    matches = []
    for box in boxes:
        face = crop_image(image, box)
    
        recog = DeepFace.find(img_path = face, db_path = "./my_db", enforce_detection=False, silent=True)

        #TODO find names and attach to boxes
        # loop over the facial embeddings
        name = 'Unknown'
        for item_n in range(0, len(recog[0])):
            idx = recog[0].iloc[item_n].identity.find('my_db')
            name = recog[0].iloc[item_n].identity[idx+6:]
            name = name[:name.find('/')]
            
        matches.append([box, name])

    return matches
    

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("[INFO] detecting faces...")

detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=0.5,
                            overlap_thr=0.7)

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]


try:
    while True:
        start_time = time.time()
        conectado, image = camera.read()

        boxes, scores = detector.inference(image)

        matches = recognize_faces(image, boxes)
    
        # loop over the recognized faces
        for ((x_min, y_min, x_max, y_max), name) in matches:
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