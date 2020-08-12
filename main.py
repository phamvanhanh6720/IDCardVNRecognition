from cropper.cropper import Cropper
from detector.detector import Detector
from core.utils import preprocess_image, draw_bbox
import tensorflow as tf
import cv2
import grpc
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time

if __name__ == '__main__':

    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    # model_name
    request.model_spec.name = "cropper_model"
    # signature name, default is 'serving_default'
    request.model_spec.signature_name = "serving_default"
    start = time.time()
    img, original_image, original_width, original_height = preprocess_image('test_crop.jpg', Cropper.TARGET_SIZE)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img, dtype=np.float32, shape=img.shape))

    try:
        result = stub.Predict(request, 10.0)
        result = result.outputs["tf_op_layer_concat_14"].float_val
        result = np.array(result).reshape((-1, 9))

    except Exception as e:
        print(e)
    end = time.time()
    print("time_1:{}".format(end-start))

    cropper = Cropper()
    cropper.set_best_bboxes(result, original_width=original_width, original_height=original_height, iou_threshold=0.5)
    cropper.set_image(original_image=original_image)

    aligned_image = getattr(cropper, "image_output")
    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

    original_height, original_width, _ = aligned_image.shape
    img = cv2.resize(aligned_image, Detector.TARGET_SIZE)

    img, original_image, original_width, original_height = preprocess_image('cropped.jpg', Cropper.TARGET_SIZE)
    # model_name
    request.model_spec.name = "detector_model"
    # signature name, default is 'serving_default'
    request.model_spec.signature_name = "serving_default"

    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img, dtype=np.float32, shape=img.shape))

    try:
        result = stub.Predict(request, 10.0)
        result = result.outputs["tf_op_layer_concat_14"].float_val
        result = np.array(result).reshape((-1, 13))

    except Exception as e:
        print(e)

    detector = Detector()
    detector.set_best_bboxes(result, original_width=original_width, original_height=original_height, iou_threshold=0.5)
    detector.set_info_images(original_image=original_image)
    drawn_bbox = draw_bbox(np.copy(original_image), getattr(detector, "best_bboxes"))
    cv2.imwrite()
    info_images = getattr(detector, "info_images")
    keys = info_images.keys()
    for key in keys:
        infor = info_images[key]

        for i in range(len(infor)):
            cv2.imwrite(key + '_' + str(i) + '.jpg', infor[i]['image'])

    print("time_all: {}".format(time.time()-start))







