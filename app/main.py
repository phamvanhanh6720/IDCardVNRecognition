from app import app
from flask import render_template
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

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
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

    filepath = app.config["IMAGE_UPLOADS"]+"/"+filename
    img, original_image, original_width, original_height = preprocess_image(filepath, Cropper.TARGET_SIZE)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
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

    if not cropper.respone_client(threshold_idcard=0.5):
        return "invalid image"

    cropper.set_image(original_image=original_image)

    aligned_image = getattr(cropper, "image_output")
    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

    original_height, original_width, _ = aligned_image.shape
    img = cv2.resize(aligned_image, Detector.TARGET_SIZE)
    img = np.float32(img/255.)
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
    detector.set_info_images(original_image=aligned_image)
    info_images = getattr(detector, "info_images")


    keys = info_images.keys()
    plt.imsave(app.config["IMAGE_UPLOADS"]+"/crop.jpg", info_images['full_name'][0]['image'])

    img = app.config["IMAGE_UPLOADS"]+"/crop.jpg"
    img = Image.open(img)
    s = reader.predict(img)
    print("total_time:{}".format(time.time()-start))
    return s

