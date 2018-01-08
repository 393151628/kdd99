from analysis import create_app
from config import ENV

from gevent import monkey
from gevent.pywsgi import WSGIServer

# monkey.patch_all()
from urls import register_url

if __name__ == '__main__':
    app = create_app(ENV)
    register_url(app)
    app.run(host='0.0.0.0', debug=True)
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
