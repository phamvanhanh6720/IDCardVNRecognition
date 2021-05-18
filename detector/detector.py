import cv2
import numpy as np
from typing import List, Dict, Any, Union


class Detector:

    def __init__(self, config_path, weight_path, score_threshold=0.5, iou_threshold=0.6):

        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.i2label_cc = {0: 'id', 1: 'full_name', 2: 'date_of_birth', 3: 'sex', 4: 'nationality',
                           5: 'nation', 6: 'address_info', 7: 'portrait', 8: 'duration', 9: 'place_of_birth',
                           10: 'place_of_residence'}

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

    @staticmethod
    def crop(original_image, b_box):
        x_min, y_min, x_max, y_max, _, _ = list(map(int, list(b_box)))
        cropped_image = original_image[y_min:y_max, x_min: x_max]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        return cropped_image

    @staticmethod
    def decode_address(address_bbox: np.ndarray):
        """
        address_info[i] = nd.array([x_min, y_min, x_max, y_max, score, class_id])
        :Return
        """

        y_mins = address_bbox[:, 1]
        args = np.argsort(y_mins)
        address_bbox = address_bbox[args]

        num_address = address_bbox.shape[0]

        address = {}

        if num_address == 4:
            address['place_of_birth'] = address_bbox[:2]
            address['place_of_residence'] = address_bbox[2:4]

            return address
        elif num_address == 2:
            address['place_of_birth'] = address_bbox[0].reshape(1, -1)
            address['place_of_residence'] = address_bbox[1].reshape(1, -1)

            return address

        bbox_1 = list(address_bbox[0])
        bbox_2 = list(address_bbox[1])
        bbox_3 = list(address_bbox[2])

        distance_12 = bbox_2[1] - bbox_1[3]
        distance_23 = bbox_3[1] - bbox_2[3]

        address['place_of_birth'] = []
        address['place_of_residence'] = []
        address['place_of_birth'].append(bbox_1)
        if distance_12 < distance_23:
            address['place_of_birth'].append(bbox_2)
        else:
            address['place_of_residence'].append(bbox_2)

        address['place_of_residence'].append(bbox_3)

        address['place_of_birth'] = np.array(address['place_of_birth'])
        address['place_of_residence'] = np.array(address['place_of_residence'])

        return address

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

        b_boxes = list()
        if len(idxs) > 0:
            for i in idxs.flatten():
                x_min, y_min, box_width, box_height = boxes[i]
                x_max = int(x_min + box_width)
                y_max = int(y_min + box_height)

                score = confidences[i]
                class_id = class_ids[i]

                # x_min, y_min, x_max, y_max, score, class
                b_boxes.append([x_min, y_min, x_max, y_max, score, class_id])

        return np.array(b_boxes)

    def select_best_bbox(self, b_boxes):
        """
        Select best bounding boxes of each class except address_info.
        In case of address_info class, choose maximum 4 bounding boxes that have higher score
        """
        classes = b_boxes[:, 5].astype(int)
        scores = b_boxes[:, 4]

        info = dict()

        for i in list(self.i2label_cc.keys()):
            mask = classes == i

            # special case: id of address_info is 6
            if i != 6 and np.sum(mask) != 0:
                idx = np.argmax(scores * mask)
                info[i] = b_boxes[idx]

            elif np.sum(mask) != 0:
                address_boxes = b_boxes[mask]
                address_scores = address_boxes[:, 4]

                if address_scores.shape[0] > 4:
                    idx = np.argsort(address_scores)
                    idx = idx[::-1]
                    address_boxes = address_boxes[idx[:4]]

                info[i] = address_boxes

        return info

    def process(self, aligned_image):

        b_boxes = self.infer_yolo(aligned_image)

        info: dict = self.select_best_bbox(b_boxes)

        info_img: Dict[Union[int, Any], Union[List[Any], Any]] = dict()

        for id in info.keys():
            if id != 6:
                info_img[id] = self.crop(aligned_image, info[id])
            elif 6 in list(info.keys()):

                address_bbox = info[6]
                address_info_dict = self.decode_address(address_bbox)
                for key in address_info_dict.keys():
                    if key == 'place_of_birth':
                        boxes = address_info_dict[key]

                        # 9 is id of place_of_birth class
                        info_img[9] = []
                        for i in range(boxes.shape[0]):
                            info_img[9].append(self.crop(aligned_image, boxes[i]))

                    elif key == 'place_of_residence':
                        boxes = address_info_dict[key]

                        # 10 is id of place_of_residence
                        info_img[10] = []
                        for i in range(boxes.shape[0]):
                            info_img[10].append(self.crop(aligned_image, boxes[i]))

        return info_img
