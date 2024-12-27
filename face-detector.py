import cv2
import time
import os.path
from libs.face_detector import FaceDetector

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
print("[INFO] detecting faces...")

detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=0.5,
                            overlap_thr=0.7)

try:
    while True:
        start_time = time.time()
        conectado, image = camera.read()

        boxes, scores = detector.inference(image)
        detector.draw_bboxes(image, boxes)
        
        cv2.putText(image, f"FPS: {1 / (time.time() - start_time):.2f}", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

        # show the output image
        cv2.imshow("Video", image)
        if cv2.waitKey(1) == ord('q'): break

finally:
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas pela OpenCV