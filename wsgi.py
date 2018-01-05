# -*- coding: utf-8 -*-
from analysis import create_app
from config import ENV
from urls import register_url

app = create_app(ENV)
register_url(app)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

