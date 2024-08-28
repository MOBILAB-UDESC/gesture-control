import threading
import socket
import time
import cv2 as cv
import cv2
import mediapipe as mp
import face_recognition
import pickle
import numpy as np
# import pyrealsense2 as rs

# Load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings/data.db", "rb").read())

# Image size for recognition
height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Define the font for text

print("[INFO] recognizing faces...")
# host = ''
# port = 9000

# locaddr = (host, port)

# # Create a UDP socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# tello_address = ('192.168.10.1', 8889)


# def recv():
#     while True:
#         try:
#             data, server = sock.recvfrom(1518)
#             print(data.decode(encoding="utf-8"))
#         except Exception:
#             print('\nExit . . .\n')
#             break


# Create and start recvThread
# recvThread = threading.Thread(target=recv)
# recvThread.start()

# Initialize Mediapipe models
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

last_sent = 0
hands_detected = False
admin_in_the_house = False
read = False
fps = 0
people = []


def detect():
    global left_hand_center, right_hand_center, center_face, hands_detected
    global people, admin_in_the_house, frame, read, fps

    both_hands_detect = 0
    fps = 0
    # Initialize face and hand detectors
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while True:
        if not read:
            continue

        image = frame # para garantir que a imagem não seja atualizada durante o processamento
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Initialize the list of names for each face detected
        people = []

        j = 0
        # Loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            people.append([name, boxes[j]])
            j = j+1

        #     # frame = cv.flip(np.asanyarray(color_frame.get_data()), 1)
        #     rgb_frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        #     # Detect faces
        #     face_results = face_detector.process(rgb_frame)
        #     frame_height, frame_width, _ = image.shape

        #     # Detect hands
        #     hand_results = hands_detector.process(rgb_frame)

        #     if face_results.detections:
        #         for face in face_results.detections:
        #             face_rect = np.multiply(
        #                 [
        #                     face.location_data.relative_bounding_box.xmin,
        #                     face.location_data.relative_bounding_box.ymin,
        #                     face.location_data.relative_bounding_box.width,
        #                     face.location_data.relative_bounding_box.height,
        #                 ],
        #                 [frame_width, frame_height, frame_width, frame_height]
        #             ).astype(int)

        #             key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
        #             key_points_coords = np.multiply(key_points, [frame_width, frame_height]).astype(int)

        #             # Calculate center of the face
        #             center_face = (int(face_rect[0] + face_rect[2] / 2), int(face_rect[1] + face_rect[3] / 2))

        #             # Draw key points on face
        #             for x, y in key_points_coords:
        #                 cv.circle(image, (x, y), 5, (255, 0, 255), -1)
        #             cv.circle(image, (center_face[0], center_face[1]), 5, (255, 0, 0), -1)

        #     if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
        #         hands_detected = True

        #         for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
        #             hand_points = []

        #             for landmark in hand_landmarks.landmark:
        #                 x = int(landmark.x * frame_width)
        #                 y = int(landmark.y * frame_height)
        #                 hand_points.append((x, y))
        #                 cv.circle(image, (x, y), 5, (255, 0, 255), cv.FILLED)

        #             # Calculate left hand center
        #             if hand_results.multi_handedness[i].classification[0].label == "Left":
        #                 left_hand_center = tuple(np.mean(hand_points, axis=0).astype(int))
        #                 cv.circle(image, (left_hand_center[0], left_hand_center[1]), 5, (255, 0, 0), cv.FILLED)

        #             # Calculate right hand center
        #             if hand_results.multi_handedness[i].classification[0].label == "Right":
        #                 right_hand_center = tuple(np.mean(hand_points, axis=0).astype(int))
        #                 cv.circle(image, (right_hand_center[0], right_hand_center[1]), 5, (255, 0, 0), cv.FILLED)
        #     else:
        #         hands_detected = False

        #     cv.putText(image, f"FPS: {fps:.2f}", (30, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        #     cv.imshow("frame", image)
        #     key = cv.waitKey(1)

        # cv.destroyAllWindows()


def control():
    global left_hand_center, right_hand_center, center_face, hands_detected

    while True:
        # Calculate hand displacements relative to the face center
        displacement_x_left = center_face[0] - left_hand_center[0]
        displacement_y_left = center_face[1] - left_hand_center[1]
        displacement_x_right = right_hand_center[0] - center_face[0]
        displacement_y_right = center_face[1] - right_hand_center[1]

        # if hands_detected:
        #     # Control drone movement based on hand positions
        #     if displacement_x_left < 60:
        #         print('Movendo o drone 70 cm para a esquerda')
        #         my_drone.move_right(70)

        #     elif displacement_x_left > 150:
        #         print('Movendo o drone 50 cm para a direita')
        #         my_drone.move_left(50)

        #     if displacement_y_right < -180:
        #         print('Move 5cm para a direita')
        #         my_drone.move_forward(50)

        #     elif displacement_y_right > 60:
        #         print('Move 5cm para a esquerda')
        #         my_drone.move_back(70)
        #     time.sleep(0.0001)


def capture():
    global read, frame, fps

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    
    frame_counter = 0
    start_time = time.time()

    while True:
        read, frame = camera.read()
        if not read:
            continue

        frame_counter += 1
        fps = frame_counter / (time.time() - start_time)

# Create and start captureThread
captureThread = threading.Thread(target=capture)
captureThread.start()

# Create and start detectThread
camThread = threading.Thread(target=detect)
camThread.start()

control_started = False
admins = ['deborah', 'douglas']

while True:
    if not read:
        continue

    image = frame
    if not admin_in_the_house:
        for person in people:
            for name, (y1, x1, y2, x2) in people:
                # Desenhando retangulo da face
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, name, (x2, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        cv2.putText(image, f"FPS: {fps:.2f}", (30, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("frame", image) 
        cv2.waitKey(1)        


    # for name in names:
    #     if name in admins: 
    #         if not admin_in_the_house:
    #             print('ACHEI' + name)
    #             admin_in_the_house = True
    #             break
    #     else:
    #         print('Corram o Admin sumiu!!!!')
    #         admin_in_the_house = False

    # if hands_detected and not control_started:
    #     controlThread = threading.Thread(target=control)
    #     controlThread.start()
    #     control_started = True