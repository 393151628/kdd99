# -*- coding: utf-8 -*-
from analysis import create_app
from config import ENV

app = create_app(ENV)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

