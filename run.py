from analysis import create_app
from config import ENV

if __name__ == '__main__':
    app = create_app(ENV)
    app.run(host='0.0.0.0', debug=True)
