from cropper.cropper import Cropper
from core.utils import preprocess_image, draw_bbox
import tensorflow as tf
import cv2
import os
import numpy as np

if __name__=='__main__':
    detector_model_path = os.path.join('models', 'cropper', '1')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    detector_model = tf.saved_model.load(detector_model_path)


    # Chạy để crop ra căn cước
    # Thư mục images chứa toàn bộ ảnh cần crop, ko để lên github vì dung lượng
    # lớn, tải trên drive

    lst_images = os.listdir('images')
    file = open('invalid.txt', "w")
    for i in range(len(lst_images)):
        img, original_image, original_width, orignal_height = preprocess_image('images/' + lst_images[i], Cropper.TARGET_SIZE)

        pred = detector_model(img)
        best_bboxes = Cropper.decode_prediction(pred, original_width, orignal_height, 0.5)

        if Cropper.respone_client(best_bboxes, 0.6):
            points = Cropper.convert_bbox_to_points(best_bboxes)
            drawn_image = draw_bbox(np.copy(original_image), best_bboxes)

            aligned_image = Cropper.align_image(np.copy(original_image), points)

            cv2.imwrite('aligned/' + 'aligned' + str(i)+'.jpg', aligned_image)
        else:
            file.write("images/" + lst_images[i] + "\n")

    file.close()