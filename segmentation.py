import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11n-seg.pt")  # segmentation model

WEBCAM_RAW_RES = [(1280, 720), (960, 540), (848, 480), (640, 480)]
RES = 1

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RAW_RES[RES][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RAW_RES[RES][1])
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

cv2.namedWindow(winname='instance-segmentation-object-tracking')

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True,  verbose=False)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
   
    if im0 is not None:
        cv2.imshow("instance-segmentation-object-tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()