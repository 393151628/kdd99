# -*- coding: utf-8 -*-
from flask import Blueprint
receive_blueprint = Blueprint('receive_data', __name__)

from . import view
