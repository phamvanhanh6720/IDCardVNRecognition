import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

if __name__ == '__main__':
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    img = '/home/phamvanhanh/PycharmProjects/ComputerVison/IDCardVNRecognition/Test_Opencv/test_12.png'
    img = Image.open(img)
    plt.imshow(img)
    s = detector.predict(img)

    print(s)