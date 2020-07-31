import tensorflow as tf
import numpy as np
from core.utils import nms
import cv2
import matplotlib.pyplot as plt
from core.utils import draw_bbox
import os

def preprocess_image(image_path, target_size):

    img = tf.keras.preprocessing.image.load_img(image_path)
    original_width, original_height = img.size

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)/255.

    # convert ndarray to Tensorflow tensor
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)

    return img, original_width, original_height



def decode_prediction(pred, original_width, original_height, iou_threshold):
    pred = tf.reshape(pred, (tf.shape(pred).numpy()[1], tf.shape(pred).numpy()[2]))
    #convert Tensorflow tensor to ndarray
    pred = tf.make_tensor_proto(pred)
    pred = tf.make_ndarray(pred)

    # coordinates[i] : (y_min, x_min, y_max, x_max)
    coordinates = pred[:, 0:4]
    y_mins = coordinates[:, 0:1]*original_height
    x_mins = coordinates[:, 1:2]*original_width
    y_maxs = coordinates[:, 2:3]*original_height
    x_maxs = coordinates[:, 3:4]*original_width

    scores = pred[:, 4:9]
    classes = np.argmax(scores, axis=-1)
    classes = np.expand_dims(classes, axis=-1)
    scores = np.max(scores, axis=-1, keepdims=True)

    # bboxes : (xmin, ymin, xmax, ymax, score, class)
    bboxes = np.hstack((x_mins, y_mins, x_maxs, y_maxs, scores, classes))
    best_bboxes = nms(bboxes, iou_threshold=iou_threshold)

    return best_bboxes



if __name__=='__main__':
    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    img, original_width, original_height = preprocess_image('test4.jpg', (416, 416))

    detector_model_path = os.path.join('models', 'detector', '1')

    model = tf.saved_model.load(detector_model_path)
    pred = model(img)

    best_bboxes = decode_prediction(pred, original_width=original_width, original_height=original_height, iou_threshold=0.5)

    best_bboxes = np.array(best_bboxes)


    print(best_bboxes)
    image = cv2.imread('test4.jpg')
    img_draw = draw_bbox(image, best_bboxes)
    """
    img_draw =cv2.resize(img_draw, (image.shape[1]//4, image.shape[0]//4))
    cv2.imshow('result', img_draw)
    cv2.waitKey(0)"""
    cv2.imwrite('result4.jpg', img_draw)