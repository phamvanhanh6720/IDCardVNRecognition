from cropper.cropper import Cropper
from detector.detector import Detector
from core.utils import preprocess_image, draw_bbox
import tensorflow as tf
import cv2
import os
import numpy as np

if __name__=='__main__':
    cropper_model_path = os.path.join('models', 'cropper', '1')
    detector_mode_path = os.path.join('models','detector','1')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    cropper_model = tf.saved_model.load(cropper_model_path)
    img, original_image, original_width, original_height = preprocess_image('test_real.jpg', Cropper.TARGET_SIZE)
    cropper = Cropper()
    pred = cropper_model(img)
    cropper.set_best_bboxes(pred, original_height=original_height, original_width=original_width, iou_threshold=0.5)
    cropper.set_image(original_image=original_image)
    cv2.imwrite('result.jpg', getattr(cropper, 'image_output'))

    del cropper
    img, original_image, original_width, original_height = preprocess_image('result.jpg', Cropper.TARGET_SIZE)

    detector_model = tf.saved_model.load(detector_mode_path)

    detector = Detector()
    pred = detector_model(img)

    detector.set_best_bboxes(pred, original_width=original_width, original_height=original_height, iou_threshold=0.5)
    detector.set_info_images(original_image)
    info_images = getattr(detector, "info_images")
    keys = info_images.keys()
    for key in keys:
        infor = info_images[key]

        for i in range(len(infor)):
            cv2.imwrite(key + '_' + str(i) + '.jpg', infor[i]['image'])




    """    images = os.listdir('aligned/')

    for image in images:
        img, original_image, original_width, original_height = preprocess_image('aligned/'+image, Cropper.TARGET_SIZE)
        pred = detector_model(img)
        detector = Detector()

        detector.set_best_bboxes(pred, original_width=original_width, original_height=original_height, iou_threshold=0.5)
        detector.set_info_images(original_image)

        info_images = getattr(detector, "info_images")
        keys = info_images.keys()
        for key in keys:
            infor = info_images[key]

            for i in range(len(infor)):
                cv2.imwrite('info_images/'+image.split('.')[0]+'_'+key+'_'+str(i)+'.jpg', infor[i]['image'])

        del detector"""




