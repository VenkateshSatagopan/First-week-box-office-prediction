import shutil
import sys

from flask import Flask, send_from_directory
from flask import jsonify
from flask import request
from flask_cors import CORS
from waitress import serve

shutil.rmtree('../models/metadata/webdriver/temp')
from ModelHandler import ModelHandler
from threading import Thread

app = Flask(__name__)

# use cors to accept Cross-Origin Requests
# this allows requests from a SPA served separately, for example from npm run serve
# makes development more streamlined
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
cors = CORS(app)


handler = ModelHandler()
t = Thread(target=handler)
t.start()

@app.route('/api/<token>')
def api_resp(token):
    if token == str(-1):
        result = {}
    else:
        result = handler.ask(token)
    return jsonify(result)


@app.route('/api', methods=['POST'])
def new_request():
    # token = handler.submit('IMDB', {'IMDB': 'tt4154796'})
    # token = handler.submit('YT', {'YT': 'https://www.youtube.com/watch?v=2NwHpkEjn84'})
    # token = handler.submit('BOTH', {'IMDB': 'tt3480822', 'YT': 'https://www.youtube.com/watch?v=2NwHpkEjn84'})
    TYPE = request.form['TYPE']
    IMDB = request.form['IMDB']
    YT = request.form['YT']

    if TYPE == 'IMDB':
        token = handler.submit('IMDB', {'IMDB': IMDB})
    elif TYPE == 'YT':
        token = handler.submit('YT', {'YT': 'https://www.youtube.com/watch?v=' + YT})
    elif TYPE == 'BOTH':
        token = handler.submit('BOTH', {'IMDB': IMDB, 'YT': 'https://www.youtube.com/watch?v=' + YT})
    else:
        token = -1
    return str(token)


@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory('../frontend2/dist', path)


@app.route('/')
def root():
    return send_from_directory('../frontend2/dist', 'index.html')


@app.errorhandler(500)
def server_error(e):
    return 'An internal error occurred [server.py] %s' % e, 500


if __name__ == '__main__':
    # This is used when running locally only. When deploying use a web-server process
    # such as waitress to serve the app.
    try:
        set_port = (sys.argv[1])
    except IndexError:
        set_port = 80

    # app.run(host='0.0.0.0', port = set_port, debug=True, threaded=True)

    # Serve with waitress
    serve(app, host='0.0.0.0', port=set_port)
