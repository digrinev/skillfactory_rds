import os
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = '../models/research/deploy/pipeline_file.config'
PATH_TO_CKPT = '../training/'
PATH_TO_LABELS = '../data/label_map.pbtxt'

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Init our model configs
def init_tf_model():
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-100')).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    return detection_model, category_index

# Detect function
@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def main():
    # Load our model
    detection_model, category_index = init_tf_model()

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    videoWriter = cv2.VideoWriter('video.avi', fourcc, 15.0, (640,480))

    while True:
        # Read frame from camera
        ret, image_np = cap.read()

        # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor, detection_model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        # Write Video
        videoWriter.write(image_np_with_detections)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



