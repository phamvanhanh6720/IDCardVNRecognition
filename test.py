from cropper import Cropper
import cv2


if __name__ == '__main__':
    cfg_path = '/home/phamvanhanh/PycharmProjects/ComputerVison/IDCardVNRecognition/cropper/yolov4_tiny.cfg'
    weight_path = '/home/phamvanhanh/PycharmProjects/ComputerVison/IDCardVNRecognition/cropper/yolov4_tiny_final.weights'
    cropper = Cropper(config_path=cfg_path, weight_path=weight_path)
    img_path = '/media/phamvanhanh/3666AAB266AA7273/DATASET/Dataset_CCCD/cmt+cccd/d522439536d6c6889fc781.jpg'

    is_id_card = cropper.process(filepath=img_path)
    if is_id_card:
        aligned = getattr(cropper, "aligned_image")
        cv2.imshow('test', aligned)
        cv2.waitKey(0)
    else:
        print("It is not id card")