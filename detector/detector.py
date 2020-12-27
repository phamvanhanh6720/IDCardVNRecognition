import tensorflow as tf
import numpy as np
import cv2
from core.utils import  nms
from tensorflow_serving.apis import predict_pb2

class Detector:
    TARGET_SIZE = (416, 416)

    def __init__(self, stub, aligned_image, iou_threshold=0.5):
        self.info_images = None
        self.best_bboxes = None

        self.stub = stub
        self.original_height, self.original_width, _ = aligned_image.shape
        self.aligned_image = aligned_image
        self.iou_threshold = iou_threshold


    def request_server(self):

        # preprocess image before request
        img = cv2.resize(self.aligned_image, self.TARGET_SIZE)
        img = np.float32(img/255.)
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        request = predict_pb2.PredictRequest()
        # model_name
        request.model_spec.name = "detector_model"
        request.model_spec.signature_name = "serving_default"

        # new request to detector model
        request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img, dtype=np.float32, shape=img.shape))
        try:
            result = self.stub.Predict(request, 10.0)
            result = result.outputs["tf_op_layer_concat_14"].float_val
            result = np.array(result).reshape((-1, 13))

        except Exception as e:
            print(e)

        return result

    def process(self):
        response = self.request_server()
        final_best_bboxes = self.decode_prediction(response, original_width=self.original_width, original_height=self.original_height,
                                      iou_threshold=self.iou_threshold)

        setattr(self, "best_bboxes", final_best_bboxes)
        self.reponse_client()

        infor_images = self.process_info_images(self.aligned_image)
        # print(infor_images)
        setattr(self, "infor_images", infor_images)

    def reponse_client(self):
        """
        Check Aligned image is new id card or old id card
        If Old ID Card:
            Response Invalid Image
        """
        classes = self.best_bboxes[:, 5].astype(int)
        scores = self.best_bboxes[:, 4]

        # 2: idx of date_of_birth class
        mask = classes == 2
        if sum(mask) == 0:
            raise Exception("Aligned Image is old id card")

        # check position of id box and full_name box
        # 0: idx of id class
        mask = classes == 0
        idxs = np.where(mask==True)[0]
        if len(idxs) ==0:
            raise Exception("Cannot find id box in aligned image")
        idx_tmp = np.argsort(scores[idxs])[::-1][0]
        id_box = (self.best_bboxes[idxs])[idx_tmp]

        # 1: idx of id class
        mask = classes == 1
        idxs = np.where(mask==True)[0]
        if len(idxs) == 0:
            raise Exception("Cannot find full_name box in aligned image")
        idx_tmp = np.argsort(scores[idxs])[::-1][0]
        fullname_box = (self.best_bboxes[idxs])[idx_tmp]

        if not(id_box[1] < fullname_box[1]):
            raise Exception("Position of full_name box and id box are not correct")


    def decode_prediction(self, pred, original_width, original_height, iou_threshold):
        """
        :param pred: ndarray 2-D : respone of detector model
        :param original_width:
        :param original_height:
        :param iou_threshold:
        :return: ndarray best_bboxes: (x_min, y_min, x_max, y_max, score, class)
        label=> index:
        id                  0
        full_name           1
        data_of_birth       2
        sex                 3
        quoc_tich           4
        dan_toc             5
        address_info        6
        chan_dung           7
        thoi_han            8
        """

        # coordinates[i] : (y_min, x_min, y_max, x_max)
        coordinates = pred[:, 0:4]
        y_mins = coordinates[:, 0:1] * original_height
        x_mins = coordinates[:, 1:2] * original_width
        y_maxs = coordinates[:, 2:3] * original_height
        x_maxs = coordinates[:, 3:4] * original_width

        scores = pred[:, 4:13]
        classes = np.argmax(scores, axis=-1)
        classes = np.expand_dims(classes, axis=-1)
        scores = np.max(scores, axis=-1, keepdims=True)

        # bboxes : (xmin, ymin, xmax, ymax, score, class)
        bboxes = np.hstack((x_mins, y_mins, x_maxs, y_maxs, scores, classes))
        best_bboxes = nms(bboxes, iou_threshold=iou_threshold)
        best_bboxes = np.array(best_bboxes)

        # the maximum bboxes of address_info is 4 bboxes
        classes = best_bboxes[:, 5].astype(int)
        scores = best_bboxes[:, 4]

        mask = classes == 6
        if sum(mask) <= 4:
            return best_bboxes

        addr_idxs = np.where(mask == True)[0]
        idxs = np.where(mask != True)[0]

        non_addr_boxes = best_bboxes[idxs]
        addr_boxes = best_bboxes[addr_idxs]

        best_addr_boxes = addr_boxes[np.argsort(scores[idxs])[::-1][:4]]
        final_best_bboxes = np.concatenate((non_addr_boxes, best_addr_boxes), axis=0)

        return final_best_bboxes

    def decode_infor(self):
        classes = self.best_bboxes[:, 5].astype(int)
        label = {'0': 'id', '1': 'full_name', '2': 'date_of_birth', '3': 'sex', '4': 'quoc_tich'
            , '5': 'dan_toc', '6': 'address_info', '7': 'chan_dung', '8': 'thoi_han'}
        infor = {}
        for i in range(len(classes)):
            key = label[str(classes[i])]
            if key not in infor.keys():
                infor[key] = []
                infor[key].append(list(self.best_bboxes[i][:5]))
            else:
                infor[key].append(list(self.best_bboxes[i][:5]))

        if "address_info" in infor.keys():
            address_info = infor['address_info']
            infor.pop("address_info")
            dict_address = self.decode_address(address_info)

            infor['que_quan'] = dict_address['que_quan']
            infor['noi_thuong_tru'] = dict_address['noi_thuong_tru']

        return infor

    def decode_address(self, address_info):
        """
        :param address_info: list of lists, address_info[i]=[x_min, y_min, x_max, y_max, score]
        """
        address_info = np.asarray(address_info)
        y_mins = address_info[:, 1]
        args = np.argsort(y_mins)
        address_info = address_info[args]

        num_address = address_info.shape[0]
        dict_address = {}

        if num_address == 4:
            dict_address['que_quan'] = [list(address_info[0]), list(address_info[1])]
            dict_address['noi_thuong_tru'] = [list(address_info[2]), list(address_info[3])]
            return dict_address
        elif num_address == 2:
            dict_address['que_quan'] = [list(address_info[0])]
            dict_address['noi_thuong_tru'] = [list(address_info[1])]
            return dict_address

        bbox_1 = list(address_info[0])
        bbox_2 = list(address_info[1])
        bbox_3 = list(address_info[2])

        distance_12 = bbox_2[1] - bbox_1[3]
        distance_23 = bbox_3[1] - bbox_2[3]

        dict_address['que_quan'] = []
        dict_address['noi_thuong_tru'] = []
        dict_address['que_quan'].append(bbox_1)
        if distance_12 < distance_23:
            dict_address['que_quan'].append(bbox_2)
        else:
            dict_address['noi_thuong_tru'].append(bbox_2)

        dict_address['noi_thuong_tru'].append(bbox_3)
        return dict_address

    def crop_image(self, original_image, x_min, y_min, x_max, y_max):
        cropped_image = original_image[y_min:y_max, x_min: x_max]
        return cropped_image

    def process_info_images(self, original_image):
        original_height, original_width, _ = original_image.shape
        infor = self.decode_infor()
        keys = infor.keys()

        infor_images = {}

        for key in keys:
            infor_images[key] = []
            for i in range(len(infor[key])):
                bbox_coor = infor[key][i][:4]
                score = infor[key][i][4]

                cropped_image = self.crop_image(original_image, int(bbox_coor[0]), int(bbox_coor[1]), int(bbox_coor[2]), int(bbox_coor[3]))
                infor_images[key].append({'image': cropped_image, 'score': score})

        return infor_images


