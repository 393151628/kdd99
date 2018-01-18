# -*- coding: utf-8 -*-
from datetime import datetime
from flask_mongoengine import MongoEngine
import time

db = MongoEngine()


class Flow(db.Document):
    dip = db.StringField()
    dport = db.StringField()
    sip = db.StringField()
    sport = db.StringField()
    error_type = db.StringField()
    error_per = db.StringField()
    timestamp = db.StringField()
    createdtime = db.StringField()

    meta = {
        'collection': 'flow',
        'index_background': True
    }
