# -*- coding: utf-8 -*-
from flask_mongoengine import MongoEngine

db = MongoEngine()


class Flow(db.Document):
    dip = db.StringField()
    dport = db.StringField()
    sip = db.StringField()
    sport = db.StringField()
    error_type = db.StringField()
    error_per = db.StringField()
    timestamp = db.StringField()

    meta = {
        'collection': 'flow',
        'index_background': True
    }
