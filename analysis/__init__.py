# -*- coding: utf-8 -*-
from flask import Flask

from analysis.models import db
from config import configs


def create_app(env, name=None):
    if name:
        app = Flask(name)
    else:
        app = Flask(__name__)
    app.config.from_object(configs[env])
    db.init_app(app)

    return app
