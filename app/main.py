from app import app

from flask import render_template, request, redirect, url_for
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
from PIL import Image

from reader.reader import  Predictor
from vietocr.tool.config import Cfg
"""
=========================
== Reader model
=========================
"""
config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = '/home/phamvanhanh/PycharmProjects/weights_transformerocr/transformerocr_v7.pth'
config['weights'] = 'https://drive.google.com/uc?export=download&id=1EaftcBJrVNDHHBnnoX1H-5n0IRX5JjCM'
# config['weights'] = './models/reader/transformerocr_best.pth'
config['weights'] = 'https://drive.google.com/uc?export=download&id=1-olev206xLgXYf7rnwHrcZLxxLg5rs0p'
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
reader = Predictor(config)


@app.route("/")
def index():
    return redirect(url_for("upload_image"))

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]


def allowed_image(filename):

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def reorient_image(im):
    im = Image.open(im)
    try:
        image_exif = im._getexif()
        image_orientation = image_exif[274]
        print(image_orientation)
        if image_orientation in (2, '2'):
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation in (3, '3'):
            return im.transpose(Image.ROTATE_180)
        elif image_orientation in (4, '4'):
            return im.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (5, '5'):
            return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (6, '6'):
            return im.transpose(Image.ROTATE_270)
        elif image_orientation in (7, '7'):
            return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (8, '8'):
            return im.transpose(Image.ROTATE_90)
        else:
            return im
    except (KeyError, AttributeError, TypeError, IndexError):
        return im


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print(type(image))
            if image.filename == "":
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                reoriented_img = reorient_image(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                reoriented_img.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                return redirect(url_for("predict", filename=filename))

            else:
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
    try:
        result = stub.Predict(request, 10.0)
        result = result.outputs["tf_op_layer_concat_14"].float_val
        result = np.array(result).reshape((-1, 9))

    except Exception as e:
        print(e)

    cropper = Cropper()
    cropper.set_best_bboxes(result, original_width=original_width, original_height=original_height, iou_threshold=0.5)

    # respone to client if image is invalid
    if not cropper.respone_client(threshold_idcard=0.8):
        return render_template('upload_image_again.html')

    cropper.set_image(original_image=original_image)

    # output of cropper part
    aligned_image = getattr(cropper, "image_output")
    cv2.imwrite('app/static/aligned_images/' + filename, aligned_image)
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

    # init default value of quoc_tich, dan_toc
    infors['quoc_tich'] = ""
    infors['dan_toc'] = ""

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
    que_quan_0 = infors['que_quan'][0]
    que_quan_1 = ''
    noi_thuong_tru_0 = infors['noi_thuong_tru'][0]
    noi_thuong_tru_1 = ''
    if len(infors['que_quan']) == 2:
        que_quan_1 = infors['que_quan'][1]
    if len(infors['noi_thuong_tru']) == 2:
        noi_thuong_tru_1 = infors['noi_thuong_tru'][1]        

    print("total_time:{}".format(time.time()-start))
    return render_template('predict.html', id=infors['id'][0].replace(" ",""), full_name=infors['full_name'][0],
                            date_of_birth=infors['date_of_birth'][0],
                            sex=infors['sex'][0],
                            quoc_tich=infors['quoc_tich'],
                            dan_toc=infors['dan_toc'],
                            que_quan_0=que_quan_0,
                            que_quan_1=que_quan_1,
                            noi_thuong_tru_0=noi_thuong_tru_0,
                            noi_thuong_tru_1=noi_thuong_tru_1,
                            filename=str(filename))
