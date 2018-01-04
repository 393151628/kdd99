# -*- coding: utf-8 -*-
from analysis import create_app
from analysis import models
from analysis.models import db
from flask_script import Manager, Shell

from config import ENV

app = create_app(ENV)

manager = Manager(app)


def _make_context():
    return dict(app=app, db=db, models=models)


manager.add_command("shell", Shell(make_context=_make_context, use_ipython=True))
if __name__ == "__main__":
    manager.run()
