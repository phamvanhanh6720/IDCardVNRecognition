import numpy as np
from core.utils import nms
import cv2
from core.utils import preprocess_image, draw_bbox
import os
from app import app
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

class Cropper:

    TARGET_SIZE = (416, 416)
    IMAGE_SIZE = (1920, 1200)
    def __init__(self, stub, filename, iou_threshold=0.5, threshold_idcard=0.8):

        self.stub = stub
        self.filepath = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        self.iou_threshold = iou_threshold
        self.threshold_idcard = threshold_idcard
        self.aligned_image = None
        self.best_bboxes = None
        self.choose_image = None

        self.image_0 = cv2.imread(self.filepath)
        self.height, self.width, _ = self.image_0.shape

        # Rotate 90
        self.image_90 = cv2.rotate(self.image_0, cv2.ROTATE_90_CLOCKWISE)

        # Rotate 180
        self.image_180 = cv2.rotate(self.image_0, cv2.ROTATE_180)

        # Rotate 270
        self.image_270 = cv2.rotate(self.image_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def request_server(self):
        request = predict_pb2.PredictRequest()
        # model_name
        request.model_spec.name = "cropper_model"
        # signature name, default is 'serving_default'
        request.model_spec.signature_name = "serving_default"

        img_1 = preprocess_image(self.image_0, self.TARGET_SIZE)
        img_2 = preprocess_image(self.image_90, self.TARGET_SIZE)
        img_3 = preprocess_image(self.image_180, self.TARGET_SIZE)
        img_4 = preprocess_image(self.image_270, self.TARGET_SIZE)

        # request to cropper model
        # img_1
        request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img_1, dtype=np.float32, shape=img_1.shape))
        try:
            result_1 = self.stub.Predict(request, 10.0)
            result_1 = result_1.outputs["tf_op_layer_concat_14"].float_val
            result_1 = np.array(result_1).reshape(-1, 9)
        except Exception as e:
            print(e)

        # img_2
        request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img_2, dtype=np.float32, shape=img_2.shape))
        try:
            result_2 = self.stub.Predict(request, 10.0)
            result_2 = result_2.outputs["tf_op_layer_concat_14"].float_val
            result_2 = np.array(result_2).reshape(-1, 9)
        except Exception as e:
            print(e)

        # img_3
        request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img_3, dtype=np.float32, shape=img_3.shape))
        try:
            result_3 = self.stub.Predict(request, 10.0)
            result_3 = result_3.outputs["tf_op_layer_concat_14"].float_val
            result_3 = np.array(result_3).reshape(-1, 9)
        except Exception as e:
            print(e)

        # img_4
        request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img_4, dtype=np.float32, shape=img_4.shape))
        try:
            result_4 = self.stub.Predict(request, 10.0)
            result_4 = result_4.outputs["tf_op_layer_concat_14"].float_val
            result_4 = np.array(result_4).reshape(-1, 9)
        except Exception as e:
            print(e)

        response = [result_1, result_2, result_3, result_4]

        return response

    def decode_total(self, response, iou_threshold=0.5):
        idx = 10

        for i in range(4):
            if i % 2 == 0:
                width, height = self.width, self.height
                if self.decode_prediction(response[i], original_width=width, original_height=height, iou_threshold=iou_threshold):
                    idx = i

            else:
                width, height = self.height, self.width
                if self.decode_prediction(response[i], original_width=width, original_height=height, iou_threshold=iou_threshold):
                    idx = i
        if idx == 10:
            raise Exception("Image is Invalid")
        if idx == 0:
            setattr(self, "choose_image", self.image_0)
        if idx == 1:
            setattr(self, "choose_image", self.image_90)
        if idx == 2:
            setattr(self, "choose_image", self.image_180)
        if idx == 3:
            setattr(self, "choose_image", self.image_270)

    def decode_prediction(self, pred, original_width, original_height, iou_threshold):
        """
        :param pred: ndarray 2-D : respone of cropper model
        :param original_width:
        :param original_height:
        :param iou_threshold:
        :return: ndarray best_bboxes: (x_min, y_min, x_max, y_max, score, class)
        """

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

        num_objs = [0 for i in range(5)]
        for i in range(len(best_bboxes)):
            class_idx = int(best_bboxes[i, 5])
            num_objs[class_idx] += 1

        # check image whether contains 5 classes
        if 0 in num_objs:
            return False
        else:
            # select best box of each class
            final_best_bboxes = np.zeros((5, best_bboxes.shape[1]))
            classes = best_bboxes[:, 5].astype(int)
            scores = best_bboxes[:, 4]

            for i in range(5):
                mask = classes == i
                idx = np.argmax(scores * mask)
                final_best_bboxes[i] = best_bboxes[idx]

            setattr(self, "best_bboxes", final_best_bboxes)

            return True

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

        idx = list(np.where((self.best_bboxes[:, 5]).astype(int) == 4))

        if not idx:
            raise Exception("IDCard Scores is lower than threshold")
        else:
            id_card_box = self.best_bboxes[idx]
            id_card_score = id_card_box[0, 4]
            if id_card_score < threshold_idcard:
                raise Exception("IDCard Scores is lower than threshold")

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

    def process(self):
        response = self.request_server()

        # get best bounding boxes
        self.decode_total(response, iou_threshold=self.iou_threshold)

        # check score of idcard class
        self.respone_client(threshold_idcard=self.threshold_idcard)

        points = self.convert_bbox_to_points()
        aligned_image = self.align_image(self.choose_image, points=points)

        setattr(self, "aligned_image", aligned_image)







