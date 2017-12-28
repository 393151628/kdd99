# -*- coding: utf-8 -*-
from flask import Flask
from .receive_data import receive_blueprint
from .show import show_blueprint


def create_app():
    app = Flask(__name__)

    app.register_blueprint(receive_blueprint, url_prefix='/api/receive')
    app.register_blueprint(show_blueprint, url_prefix='/api/show')
    return app
