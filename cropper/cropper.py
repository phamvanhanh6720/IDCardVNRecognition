import os

import cv2
import numpy as np

from app import app


class Cropper:

    TARGET_SIZE = (416, 416)
    IMAGE_SIZE = (1920, 1200)

    def __init__(self, filename, config_path, weight_path, iou_threshold=0.5, idcard_threshold=0.8):

        self.filepath = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        self.iou_threshold = iou_threshold
        self.idcard_threshold = idcard_threshold

        self.aligned_image = None
        self.best_bboxes = None
        self.choose_image = None
        self.points = None

        # Model
        self.config_path = config_path
        self.weight_path = weight_path
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weight_path)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.image_0 = cv2.imread(self.filepath)
        self.height, self.width, _ = self.image_0.shape

        # Rotate 90
        self.image_90 = cv2.rotate(self.image_0, cv2.ROTATE_90_CLOCKWISE)

        # Rotate 180
        self.image_180 = cv2.rotate(self.image_0, cv2.ROTATE_180)

        # Rotate 270
        self.image_270 = cv2.rotate(self.image_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def preprocess_img(img):
        img = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416))

        return img

    def _infer(self):

        # preprocess image
        img_0 = self.preprocess_img(self.image_0)
        img_90 = self.preprocess_img(self.image_90)
        img_180 = self.preprocess_img(self.image_180)
        img_270 = self.preprocess_img(self.image_270)


    def decode_total(self, response, iou_threshold=0.5):
        idx = 10

        for i in range(4):
            if i % 2 == 0:
                width, height = self.width, self.height
                if self.decode_prediction(response[i], original_width=width, original_height=height,
                                          iou_threshold=iou_threshold):
                    idx = i

            else:
                width, height = self.height, self.width
                if self.decode_prediction(response[i], original_width=width, original_height=height,
                                          iou_threshold=iou_threshold):
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

    def decode_yolo(self, image: np.ndarray, threshold=0.5, nms_threshold=0.5):

        height, width, _ = image.shape
        img = self.preprocess_img(image)
        self.net.setInput(img)
        layer_outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            # loop over each of the layer output
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence >= threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image

                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")

                    x_min = int(center_x - (box_width / 2))
                    y_min = int(center_y - (box_height / 2))

                    x_max = int(center_x + (box_width / 2))
                    y_max = int(center_y + (box_height / 2))

                    boxes.append([x_min, y_min, x_max, y_max])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # nms
        idxs: np.ndarray = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=nms_threshold)

        best_b_boxes = list()
        if len(idxs) > 0:
            for i in idxs.flatten():
                x_min, y_min, x_max, y_max = boxes[i]
                score = confidences[i]
                class_id = class_ids[i]

                # x_min, y_min, x_max, y_max, score, class
                best_b_boxes.append([x_min, y_min, x_max, y_max, score, class_id])

        return best_b_boxes

    def _is_id_card(self, best_b_boxes):

        best_b_boxes: np.ndarray = np.array(best_b_boxes)
        num_objs = [0 for i in range(5)]
        for i in range(len(best_b_boxes)):
            class_idx = int(best_b_boxes[i, 5])
            num_objs[class_idx] += 1

        # check image whether contains 5 classes
        if 0 in num_objs:
            return False
        else:
            # select best box of each class
            final_best_bboxes = np.zeros((5, best_b_boxes.shape[1]))
            classes = best_b_boxes[:, 5].astype(int)
            scores = best_b_boxes[:, 4]

            for i in range(5):
                mask = classes == i
                idx = np.argmax(scores * mask)
                final_best_bboxes[i] = best_b_boxes[idx]

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

    def check_points(self):
        """
        Check points whether are correctly position
        """
        top_left, top_right, bottom_left, bottom_right = self.points
        flag = True

        # top_left
        if not (top_left[0] < top_right[0] and top_left[1] < bottom_left[1]):
            raise Exception("Top Left Point is not correctly position")

        # top_right
        if not (top_right[0] > top_left[0] and top_right[1] < bottom_right[1]):
            raise Exception("Top Right Point is not correctly position")

        # bottom_left
        if not (bottom_left[0] < bottom_right[0] and bottom_left[1] > top_left[1]):
            raise Exception("Bottom Left Point is not correctly position")

        # bottom_right
        if not (bottom_right[0] > bottom_left[0] and bottom_right[1] > top_right[1]):
            raise Exception("Top Left Point is not correctly position")

    def process(self):
        response = self._infer()

        # get best bounding boxes
        self.decode_total(response, iou_threshold=self.iou_threshold)

        # check score of idcard class
        self.respone_client(threshold_idcard=self.threshold_idcard)

        setattr(self, "points", self.convert_bbox_to_points())
        self.check_points()

        aligned_image = self.align_image(self.choose_image, points=self.points)
        setattr(self, "aligned_image", aligned_image)







