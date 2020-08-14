from app import app
from flask import render_template, jsonify
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os

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
import matplotlib.pyplot as plt

from PIL import Image

from reader.reader import  Predictor
from vietocr.tool.config import Cfg
"""
=========================
== Reader model
=========================
"""
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './models/reader/transformerocr.pth'
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
reader = Predictor(config)

@app.route("/")
def index():
    return render_template("index.html")

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                print("Image saved")

                #return render_template("upload_image.html", filename=filename)
                return redirect(url_for("predict", filename=filename))

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("upload_image.html")


@app.route("/predict/<filename>")
def predict(filename):

    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    # model_name
    request.model_spec.name = "cropper_model"
    # signature name, default is 'serving_default'
    request.model_spec.signature_name = "serving_default"
    start = time.time()
    """
    =========================================
    ===== Crop and align id card image
    =========================================
    """
    filepath = app.config["IMAGE_UPLOADS"]+"/"+filename
    # preprocess image
    img, original_image, original_width, original_height = preprocess_image(filepath, Cropper.TARGET_SIZE)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    # request to cropper model
    request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img, dtype=np.float32, shape=img.shape))
    print(filepath)
    try:
        result = stub.Predict(request, 10.0)
        result = result.outputs["tf_op_layer_concat_14"].float_val
        result = np.array(result).reshape((-1, 9))

    except Exception as e:
        print(e)

    cropper = Cropper()
    cropper.set_best_bboxes(result, original_width=original_width, original_height=original_height, iou_threshold=0.5)

    # respone to client if image is invalid
    if not cropper.respone_client(threshold_idcard=0.5):
        return "invalid image"

    cropper.set_image(original_image=original_image)

    # output of cropper part
    aligned_image = getattr(cropper, "image_output")
    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

    """
    ===========================================
    ==== Detect informations in aligned image
    ===========================================
    """
    # preprocess aligned image
    original_height, original_width, _ = aligned_image.shape
    img = cv2.resize(aligned_image, Detector.TARGET_SIZE)
    img = np.float32(img/255.)
    # model_name
    request.model_spec.name = "detector_model"
    # signature name, default is 'serving_default'
    request.model_spec.signature_name = "serving_default"

    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    # new request to detector model
    request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(img, dtype=np.float32, shape=img.shape))

    try:
        result = stub.Predict(request, 10.0)
        result = result.outputs["tf_op_layer_concat_14"].float_val
        result = np.array(result).reshape((-1, 13))

    except Exception as e:
        print(e)

    detector = Detector()
    detector.set_best_bboxes(result, original_width=original_width, original_height=original_height, iou_threshold=0.5)
    detector.set_info_images(original_image=aligned_image)
    # output of detector part
    info_images = getattr(detector, "info_images")

    """
    =====================================
    ==== Reader infors from infors image
    =====================================
    """
    keys = list(info_images.keys())
    keys.remove("thoi_han")
    keys.remove("chan_dung")
    infors = dict()
    if "quoc_tich" in keys:
        infors['quoc_tich'] = ["Việt Nam"]
        keys.remove("quoc_tich")

    if "sex" in keys:
        info_image = info_images["sex"]
        infors["sex"] = list()
        for i in range(len(info_image)):
            img = info_image[i]['image']
            s = reader.predict(img)
            if "Na" in s:
                infors["sex"].append("Nam")
            else:
                infors["sex"].append("Nữ")
        keys.remove("sex")

    if "dan_toc" in keys:
        info_image = info_images["dan_toc"]
        infors["dan_toc"] = list()
        for i in range(len(info_image)):
            img = info_image[i]['image']
            s = reader.predict(img)
            s = s.split(" ")[-1]
            infors["dan_toc"].append(s)

        keys.remove("dan_toc")

    for key in keys:
        infors[key] = list()
        info_image = info_images[key]
        for i in range(len(info_image)):
            img = info_image[i]['image']
            s = reader.predict(img)
            infors[key].append(s)

    print("total_time:{}".format(time.time()-start))
    return str(infors)

