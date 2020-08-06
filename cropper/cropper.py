import tensorflow as tf
import numpy as np
from core.utils import nms
import cv2
class Cropper:

    TARGET_SIZE = (416, 416)
    IMAGE_SIZE = (1920, 1200)
    def __init__(self):

        self.image_output = None
        self.best_bboxes = None

    @staticmethod
    def decode_prediction(pred, original_width, original_height, iou_threshold):
        """
        :param pred: Tensorflow tensor : prediction of detector model
        :param original_width:
        :param original_height:
        :param iou_threshold:
        :return: ndarray best_bboxes: (x_min, y_min, x_max, y_max, score, class)
        """
        pred = tf.reshape(pred, (tf.shape(pred).numpy()[1], tf.shape(pred).numpy()[2]))
        # convert Tensorflow tensor to ndarray
        pred = tf.make_tensor_proto(pred)
        pred = tf.make_ndarray(pred)

        # coordinates[i] : (y_min, x_min, y_max, x_max)
        coordinates = pred[:, 0:4]
        y_mins = coordinates[:, 0:1] * original_height
        x_mins = coordinates[:, 1:2] * original_width
        y_maxs = coordinates[:, 2:3] * original_height
        x_maxs = coordinates[:, 3:4] * original_width

        scores = pred[:, 4:9]
        classes = np.argmax(scores, axis=-1)
        classes = np.expand_dims(classes, axis=-1)
        scores = np.max(scores, axis=-1, keepdims=True)

        # bboxes : (xmin, ymin, xmax, ymax, score, class)
        bboxes = np.hstack((x_mins, y_mins, x_maxs, y_maxs, scores, classes))
        best_bboxes = nms(bboxes, iou_threshold=iou_threshold)
        best_bboxes = np.array(best_bboxes)

        num_bboxes = len(best_bboxes)
        if num_bboxes < 5:
            return best_bboxes

        #select best box of each class
        final_best_bboxes = np.zeros((5, best_bboxes.shape[1]))
        classes = best_bboxes[:, 5].astype(int)
        scores = best_bboxes[:, 4]

        for i in range(5):
            mask = classes == i
            idx = np.argmax(scores * mask)
            final_best_bboxes[i] = best_bboxes[idx]

        return final_best_bboxes


    def convert_bbox_to_points(self):
        """
        :param best_bboxes: ndarray shape (5, 6)
        best_bboxes[i]: (x_min, y_min, x_max, y_max, score, class)
        :return: points : list((top_left, top_right, bottom_left, bottom_right))
        """
        classes = self.best_bboxes[:, 5]
        idx = np.argsort(classes)
        top_left_box, top_right_box, bottom_left_box, bottom_right_box, id_card = self.best_bboxes[idx]

        x_top_left = int(top_left_box[0])
        y_top_left = int(top_left_box[1])
        top_left = [x_top_left, y_top_left]

        x_top_right = int(top_right_box[2])
        y_top_right = int(top_right_box[1])
        top_right = [x_top_right, y_top_right]

        x_bottom_left = int(bottom_left_box[0])
        y_bottom_left = int(bottom_left_box[3])
        bottom_left = [x_bottom_left, y_bottom_left]

        x_bottom_right = int(bottom_right_box[0] + bottom_right_box[2])//2
        y_bottom_right = int(bottom_right_box[1]+bottom_right_box[3])//2
        bottom_right = [x_bottom_right, y_bottom_right]

        points = list([top_left, top_right, bottom_left, bottom_right])
        return points


    def respone_client(self, threshold_idcard):
        num_bbxoes = self.best_bboxes.shape[0]
        if num_bbxoes < 5:
            return False

        idx = list(np.where((self.best_bboxes[:, 5]).astype(int) == 4))

        if not idx:
            return False
        else:
            id_card_box = self.best_bboxes[idx]
            id_card_score = id_card_box[0, 4]
            if id_card_score < threshold_idcard:
                return False
        return True

    def align_image(self, image, points):
        """
        :param image: ndarray of image
        :param points: list[top_left, top_right, bottom_left, bottom_right]
        :return: ndarray of aligned_image
        """
        top_left, top_right, bottom_left, bottom_right = points
        pts = np.array([top_left, top_right, bottom_right, bottom_left]).astype('float32')
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates

        height_a = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
        height_b = np.sqrt(((bottom_left[0] - top_left[0]) ** 2) + ((bottom_left[1] - top_left[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        warped = cv2.resize(warped, (1920, 1200))

        return warped
    def set_image(self, original_image):
        points = self.convert_bbox_to_points()
        self.image_output = self.align_image(original_image, points=points)
    def set_best_bboxes(self, pred, original_width, original_height, iou_threshold):
        self.best_bboxes = self.decode_prediction(pred, original_width=original_width\
                                                  , original_height=original_height, iou_threshold=iou_threshold)




