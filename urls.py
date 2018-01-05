from analysis.receive_data import receive_blueprint
from analysis.show import show_blueprint


def register_url(app):
    app.register_blueprint(receive_blueprint, url_prefix='/api/receive')
    app.register_blueprint(show_blueprint, url_prefix='/api/show')

