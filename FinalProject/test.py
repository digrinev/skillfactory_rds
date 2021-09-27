from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
import random


class FaceAligner:
    """ Modified from https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/facealigner.py
    """
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, left_eye, right_eye):

        # centers of eyes
        leftEyeCenter = left_eye
        rightEyeCenter = right_eye

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]

        angle = np.arctan(dY/dX)
        angle = (angle * 180) / np.pi

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output, decode_predictions)
from modules.anchor import decode_tf, prior_box_tf

flags.DEFINE_string('cfg_path', '/home/user/skillfactory_rds/Detector/configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '/home/user/skillfactory_rds/Detector/tests/', 'path to input image')
flags.DEFINE_integer('img_num', 10, 'number of images to proceed')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')

def main(_argv):
    # init

    face_aligner = FaceAligner()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    if not FLAGS.webcam:
        save_count = 0
        for path, subdirs, files in os.walk(FLAGS.img_path):
            
            for name in files:
                if name.endswith('.jpg'):
                    img_path = os.path.join(path, name)

                    if not os.path.exists(img_path):
                        print(f"cannot find image path from {img_path}")
                        exit()

                    if save_count < FLAGS.img_num:
                        print("[*] Processing on single image {}".format(img_path))

                        img_raw = cv2.imread(img_path)
                        img_height_raw, img_width_raw, _ = img_raw.shape
                        img = np.float32(img_raw.copy())

                        if FLAGS.down_scale_factor < 1.0:
                            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                            fy=FLAGS.down_scale_factor,
                                            interpolation=cv2.INTER_LINEAR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # pad input image to avoid unmatched shape problem
                        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

                        # run model
                        outputs = model(img[np.newaxis, ...]).numpy()

                        # recover padding effect
                        outputs = recover_pad_output(outputs, pad_params)

                        # draw and save results
                        save_img_path = os.path.join('out_' + os.path.basename(img_path))
                        
                        for prior_index in range(len(outputs)):
                            draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                                            img_width_raw)
                            cv2.imwrite(save_img_path, img_raw)
                        print(f"[*] save result at {save_img_path}")
                        save_count += 1

    else:
        cam = cv2.VideoCapture(0)

        start_time = time.time()

        while True:
            _, frame = cam.read()
            if frame is None:
                print("no cam input")

            frame_height, frame_width, _ = frame.shape

            orig_frame = frame.copy()
            
            face = None

            img = cv2.resize(frame, (512,512))
            img = np.float32(frame.copy())
            if FLAGS.down_scale_factor < 1.0:
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                fy=FLAGS.down_scale_factor,
                                interpolation=cv2.INTER_LINEAR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            start_time = time.time()
            outputs = model(img[np.newaxis, ...]).numpy()
            inference_time = f"Inf: {time.time() - start_time}"

            cv2.putText(frame, inference_time, (25, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # draw results
            for prior_index in range(len(outputs)):
                preds = decode_predictions((frame_width, frame_height), outputs)          

                for key, value in preds.items():
                    bbox = value[0]['bbox']
                    left_eye = value[0]['left_eye']
                    right_eye = value[0]['right_eye']

                    # Our face ROI
                    face = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    # Eyes
                    x1_le = left_eye[0] - 25
                    y1_le = left_eye[1] - 25
                    x2_le = left_eye[0] + 25
                    y2_le = left_eye[1] + 25

                    x1_re = right_eye[0] - 25
                    y1_re = right_eye[1] - 25
                    x2_re = right_eye[0] + 25
                    y2_re = right_eye[1] + 25

                    if left_eye[1] > right_eye[1]:
                        A = (right_eye[0], left_eye[1])
                    else:
                        A = (left_eye[0], right_eye[1])

                    # Calc our rotating degree
                    delta_x = right_eye[0] - left_eye[0]
                    delta_y = right_eye[1] - left_eye[1]
                    angle= np.arctan(delta_y/(delta_x + 1e-17))  # avoid devision by zero
                    angle = (angle * 180) / np.pi

                    # compute the desired right eye x-coordinate based on the
                    # desired x-coordinate of the left eye
                    desiredRightEyeX = 1.0 - 0.35

                    # determine the scale of the new resulting image by taking
                    # the ratio of the distance between eyes in the *current*
                    # image to the ratio of distance between eyes in the
                    # *desired* image
                    dist = np.sqrt((delta_x ** 2) + (delta_y ** 2))
                    desiredDist = (desiredRightEyeX - 0.35)
                    desiredDist *= 256
                    scale = desiredDist / dist

                    eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
			                            (left_eye[1] + right_eye[1]) // 2)

                    cv2.circle(frame, A, 5, (255, 0, 0) , -1)

                    cv2.putText(frame, str(int(angle)), (x1_le - 15, y1_le),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
                        
                    cv2.line(frame,right_eye, left_eye,(0,200,200),3)
                    cv2.line(frame,left_eye, A,(0,200,200),3)
                    cv2.line(frame,right_eye, A,(0,200,200),3)

                    cv2.line(frame,(left_eye[0], left_eye[1]), (right_eye[0], right_eye[1]),(0,200,200),3)

                    rotated = face_aligner.align(orig_frame, left_eye, right_eye)

                draw_bbox_landm(frame, outputs[prior_index], frame_height,
                                frame_width)

            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(frame, fps_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if face is not None:
                cv2.imshow('face aligned', rotated)

            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
