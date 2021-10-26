from threading import current_thread
import numpy as np
import cv2
from face_detection import RetinaFace


class FaceAligner:
    """ Modified from https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/facealigner.py
    """
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=112, desiredFaceHeight=None):
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

        angle = np.arctan(dY/(dX + 1e-17))
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


class FaceDetector:

    def __init__(self) -> None:
        self.nms_threshold = 0.4
        self.face_aligner = FaceAligner()
        self.detector = RetinaFace(gpu_id=0, model_path='/home/user/skillfactory_rds/FinalProject/models/Resnet50_Final.pth', network='resnet50')


    def detect(self, img):
        faces = self.detector([img])

        found_faces = {'boxes': [], 'landmarks': []}

        # box, landmarks, score = faces[0]
        for face in faces[0]:
            box, landmarks, score = face

            left_eye = tuple(map(int, landmarks[0]))
            right_eye = tuple(map(int, landmarks[1]))

            box = box.astype(np.int)

            if score > self.nms_threshold:
                found_faces['boxes'].append(box)
                found_faces['landmarks'].append(landmarks)
        
        return found_faces['boxes'], found_faces['landmarks']
     