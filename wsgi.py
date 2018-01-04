# -*- coding: utf-8 -*-
from analysis import create_app
from config import ENV
from celery import Celery
from celery import platforms

platforms.C_FORCE_ROOT = True
application = create_app(ENV)

celery = Celery(application.name, broker=application.config['CELERY_BROKER_URL'])
celery.conf.update(application.config)

if __name__ == '__main__':
    application.run()

