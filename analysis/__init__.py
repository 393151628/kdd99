# -*- coding: utf-8 -*-
from flask import Flask
from .receive_data import receive_blueprint


def create_app():
    app = Flask(__name__)

    app.register_blueprint(receive_blueprint, url_prefix='/api/receive')
    return app
