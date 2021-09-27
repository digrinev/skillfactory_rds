import cv2
from face_detection import RetinaFace
import numpy as np
import time
import logging
from modules.face import FaceAligner

def main():
    logging.basicConfig(filename='test.log', level=logging.DEBUG)

    # NMS
    nms_threshold = 0.4

    # Face Aligner
    face_aligner = FaceAligner()

    # ResNet50 Backbone
    detector = RetinaFace(gpu_id=0, model_path='/home/user/skillfactory_rds/Detector/models/Resnet50_Final.pth', network='resnet50')

    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()

        if frame is None:
            print("no cam input")         

        frame_height, frame_width, _ = frame.shape

        original_frame = frame.copy()

        img = np.float32(frame.copy())
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()        

        faces = detector([img])

        # box, landmarks, score = faces[0]
        for face in faces[0]:
            box, landmarks, score = face

            left_eye = tuple(map(int, landmarks[0]))
            right_eye = tuple(map(int, landmarks[1]))

            box = box.astype(np.int)

            if score > nms_threshold:

                if left_eye[1] > right_eye[1]:
                    A = (int(right_eye[0]), int(left_eye[1]))
                else:
                    A = (int(left_eye[0]), int(right_eye[1]))

                # Calc our rotating degree
                delta_x = right_eye[0] - left_eye[0]
                delta_y = right_eye[1] - left_eye[1]
                angle= np.arctan(delta_y/(delta_x + 1e-17))  # avoid devision by zero
                angle = (angle * 180) / np.pi

                cv2.circle(frame, A, 5, (255, 0, 0) , -1)

                cv2.putText(frame, str(int(angle)), (left_eye[0] - 15, left_eye[1]),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
                    
                cv2.line(frame,right_eye, left_eye,(0,200,200),3)
                cv2.line(frame,left_eye, A,(0,200,200),3)
                cv2.line(frame,right_eye, A,(0,200,200),3)

                cv2.line(frame,(left_eye[0], left_eye[1]), (right_eye[0], right_eye[1]),(0,200,200),3)

                cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2
                )

                conf = "{:.4f}".format(score)
                cv2.putText(frame, conf, (box[0], box[1]),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                rotated = face_aligner.align(original_frame, left_eye, right_eye)

        # calculate fps
        fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
        cv2.putText(frame, fps_str, (25, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        # show frame
        cv2.imshow('frame', frame)

        if len(faces) > 0:
            cv2.imshow('face aligned', rotated)
  
        if cv2.waitKey(1) == ord('q'):
            exit()

if __name__ == "__main__":
    main()