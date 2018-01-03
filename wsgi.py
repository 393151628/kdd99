# -*- coding: utf-8 -*-
from analysis import create_app
from config import ENV

application = create_app(ENV)

if __name__ == '__main__':
    application.run()

