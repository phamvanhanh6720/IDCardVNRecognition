class Config(object):
    DEBUG = False
    TESTING = False

    IMAGE_UPLOADS = "/home/phamvanhanh/PycharmProjects/IDCardVNRecognition/app/static/images"

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

    IMAGE_UPLOADS = "/home/phamvanhanh/PycharmProjects/IDCardVNRecognition/app/static/images"

class TestingConfig(Config):
    TESTING = True
