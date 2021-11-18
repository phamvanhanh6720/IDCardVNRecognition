import os
import time
import base64
import requests

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from vietocr.tool.config import Cfg
from cropper import Cropper
from detector import Detector
from reader import OCR
from .utils import download_weights, Config, Item

app = FastAPI()

dir_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(dir_path, 'static')

app.mount('/static', StaticFiles(directory=static_path), name='static')
templates = Jinja2Templates(directory=os.path.join(dir_path, 'templates'))

# Model AI
cfg = Config.load_config()

cropper = Cropper(config_path=download_weights(cfg['cropper']['cfg']),
                  weight_path=download_weights(cfg['cropper']['weight']))

detector = Detector(config_path=download_weights(cfg['detector']['cfg']),
                    weight_path=download_weights(cfg['detector']['weight']))

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = cfg['reader']['weight']
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
reader = OCR(config)


@app.post('/extract')
def extract(item: Item):
    if item.key != cfg['key_api']:
        return {'message': 'rejected'}
    else:
        img_object = base64.b64decode(item.base64_img)
        image = cv2.imdecode(np.fromstring(img_object, np.uint8), cv2.IMREAD_COLOR)

        is_card, is_id_card, warped = cropper.process(image=image)

        if is_card is False and is_id_card is None:
            return {'message': 'approved', 'description': 'please upload your id card'}

        if is_id_card is not None and warped is None:
            return {'message': 'approved', 'description': 'please upload id card again'}

        info_images = detector.process(warped)

        info = dict()

        for id in list(info_images.keys()):
            # 7 is id of portrait class
            if id == 7:
                continue
            label = detector.i2label_cc[id]
            if isinstance(info_images[id], np.ndarray):
                info[label] = reader.predict(info_images[id])
            elif isinstance(info_images[id], list):
                info[label] = []
                for i in range(len(info_images[id])):
                    info[label].append(reader.predict(info_images[id][i]))

        info['nationality'] = 'Việt Nam'
        if 'sex' in info.keys():
            if 'Na' in info['sex']:
                info['sex'] = 'Nam'
            else:
                info['sex'] = 'Nữ'

        return {'message': 'approved', 'description': 'image is id card', 'info': info}


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(os.path.join('upload_image.html'), {'request': request})


@app.post('/upload_image/')
def upload_image(request: Request, file: UploadFile = File(...)):
    if file.content_type in ['image/jpeg', 'image/png']:

        img_object = file.file.read()
        image = cv2.imdecode(np.fromstring(img_object, np.uint8), cv2.IMREAD_COLOR)

        file_name = str(time.time()) + file.filename
        print(cfg['save_image'])
        if cfg['save_image']:
            cv2.imwrite(os.path.join(dir_path, 'static', 'aligned_images', file_name), image)
        my_string = base64.b64encode(img_object)
        my_string = my_string.decode('utf-8')
        url = 'http://127.0.0.1:8000' + '/extract'
        request_body = {'base64_img': my_string, 'key': cfg['key_api']}

        response = requests.post(url=url, json=request_body)
        response = response.json()

        if response['description'] != 'image is id card':
            return templates.TemplateResponse(os.path.join('upload_image_again.html'), {'request': request})

        base64_img = "data:image/png;base64," + my_string
        return templates.TemplateResponse(os.path.join('predict.html'),
                                          {'request': request, 'base64_img': base64_img, 'info': response['info']})

    else:
        print("Fail")
        return templates.TemplateResponse(os.path.join('upload_image_again.html'), {'request': request})
