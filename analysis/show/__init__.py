# -*- coding: utf-8 -*-
from flask import Blueprint
show_blueprint = Blueprint('show', __name__)

from . import view
