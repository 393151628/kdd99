# -*- coding: utf-8 -*-
from flask import Flask

from analysis.models import db
from config import configs
from analysis.receive_data import receive_blueprint
from .show import show_blueprint


def create_app(env):
    app = Flask(__name__)
    app.config.from_object(configs[env])
    db.init_app(app)

    # app.register_blueprint(receive_blueprint, url_prefix='/api/receive')
    app.register_blueprint(show_blueprint, url_prefix='/api/show')
    return app

