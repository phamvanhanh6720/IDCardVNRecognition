class Config(object):
    DEBUG = False
    TESTING = False

    IMAGE_UPLOADS = "./app/static/images"

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

    IMAGE_UPLOADS = "./app/static/images"

class TestingConfig(Config):
    TESTING = True
