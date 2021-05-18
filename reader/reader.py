import math

import cv2
import torch
import numpy as np

from vietocr.tool.predictor import Predictor
from vietocr.tool.translate import translate, translate_beam_search


class OCR(Predictor):

    def __init__(self, config):
        super(OCR, self).__init__(config)

    def predict(self, img, return_prob=False):
        img = self.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, device=self.config['device'], dtype=torch.float)

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)

        if return_prob:
            return s, prob
        else:
            return s

    def preprocess_input(self, image: np.ndarray):

        h, w, _ = image.shape
        new_w, image_height = self.resize(w, h, self.config['dataset']['image_height'],
                                          self.config['dataset']['image_min_width'],
                                          self.config['dataset']['image_max_width'])

        img = cv2.resize(image, (new_w, image_height))
        img = np.transpose(img, (2, 0, 1))
        img = img / 255

        return img

    @staticmethod
    def resize(w, h, expected_height, image_min_width, image_max_width):
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height
