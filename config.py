class Config(object):
    DEBUG = False
    TESTING = False

    IMAGE_UPLOADS = "/home/tranhuuhuy297/Projects/IDCardVNRecognition"

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

    IMAGE_UPLOADS = "/home/tranhuuhuy297/Projects/IDCardVNRecognition"

class TestingConfig(Config):
    TESTING = True
