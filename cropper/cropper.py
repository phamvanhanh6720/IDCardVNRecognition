import cv2
import numpy as np
from typing import List

from .process_image import align_image


class Cropper:

    TARGET_SIZE = (416, 416)
    IMAGE_SIZE = (1920, 1200)

    def __init__(self, config_path, weight_path, iou_threshold=0.3, score_threshold=0.5, idcard_threshold=0.8):

        self.iou_threshold = iou_threshold
        self.idcard_threshold = idcard_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.best_bboxes = None
        # coordinate of 4 corners
        self.points = None
        self.id_score = None
        self.is_id_card = None

        # Model
        self.config_path = config_path
        self.weight_path = weight_path
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weight_path)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    @staticmethod
    def preprocess_img(img):
        img = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        return img

    def infer_yolo(self, image: np.ndarray):

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
                if confidence >= self.score_threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image

                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # nms
        idxs: np.ndarray = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.score_threshold,
                                            nms_threshold=self.iou_threshold)

        best_b_boxes = list()
        if len(idxs) > 0:
            for i in idxs.flatten():
                x_min, y_min, box_width, box_height = boxes[i]
                x_max = int(x_min + box_width)
                y_max = int(y_min + box_height)

                score = confidences[i]
                class_id = class_ids[i]

                # x_min, y_min, x_max, y_max, score, class
                best_b_boxes.append([x_min, y_min, x_max, y_max, score, class_id])

        return best_b_boxes

    def _is_chosen(self, best_b_boxes):

        best_b_boxes: np.ndarray = np.array(best_b_boxes)
        num_objs = [0 for _ in range(5)]
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
            points = self._convert_bbox_to_points()

            # check the position of corners whether is correct
            if self._check_points(points):
                setattr(self, "points", points)
            else:
                setattr(self, "points", None)
                return False

            return True

    def _convert_bbox_to_points(self) -> List[List[int]]:
        """
        :return: Coordinate of 4 corners
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

        x_bottom_right = int((bottom_right_box[0] + bottom_right_box[2])//2)
        y_bottom_right = int((bottom_right_box[1]+bottom_right_box[3])//2)
        bottom_right = [x_bottom_right, y_bottom_right]

        points = list([top_left, top_right, bottom_left, bottom_right])

        return points

    def _is_id_card(self):

        idx = list(np.where((self.best_bboxes[:, 5]).astype(int) == 4))

        if not idx:
            return False
        else:
            id_card_score = self.best_bboxes[idx[0], 4]
            # id_card_score = id_card_box[0, 4]
            setattr(self, 'id_score', id_card_score)
            if id_card_score < self.idcard_threshold:
                return False

        return True

    def _check_points(self, points):
        """
        Check points whether are correctly position
        """
        top_left, top_right, bottom_left, bottom_right = points

        # top_left
        if not (top_left[0] < top_right[0] and top_left[1] < bottom_left[1]):
            return False

        # top_right
        if not (top_right[0] > top_left[0] and top_right[1] < bottom_right[1]):
            return False
        # bottom_left
        if not (bottom_left[0] < bottom_right[0] and bottom_left[1] > top_left[1]):
            return False

        # bottom_right
        if not (bottom_right[0] > bottom_left[0] and bottom_right[1] > top_right[1]):
            return False

        return True

    def process(self, image) -> object:
        """
        Process image. Return True if image is id card. Otherwise return False
        """
        # Raw Image
        image_0 = image

        # Rotate 90
        image_90 = cv2.rotate(image_0, cv2.ROTATE_90_CLOCKWISE)

        # Rotate 180
        image_180 = cv2.rotate(image_0, cv2.ROTATE_180)

        # Rotate 270
        image_270 = cv2.rotate(image_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # inference via yolo model
        best_b_boxes_0 = self.infer_yolo(image_0)

        best_b_boxes_90 = self.infer_yolo(image_90)

        best_b_boxes_180 = self.infer_yolo(image_180)

        best_b_boxes_270 = self.infer_yolo(image_270)

        img_0_is_chosen: bool = self._is_chosen(best_b_boxes_0)
        img_90_is_chosen: bool = self._is_chosen(best_b_boxes_90)
        img_180_is_chosen: bool = self._is_chosen(best_b_boxes_180)
        img_270_is_chosen: bool = self._is_chosen(best_b_boxes_270)

        warped = None
        if img_0_is_chosen:
            warped = align_image(image_0, self.points)
        elif img_90_is_chosen:
            warped = align_image(image_90, self.points)
        elif img_180_is_chosen:
            warped = align_image(image_180, self.points)
        elif img_270_is_chosen:
            warped = align_image(image_270, self.points)

        if warped is not None:
            is_card = self._is_id_card()
            if is_card:
                setattr(self, 'is_id_card', True)
            else:
                setattr(self, 'is_id_card', False)
                return False, getattr(self, "is_id_card"), None

            return True, getattr(self, "is_id_card"), warped

        return False, None, None
