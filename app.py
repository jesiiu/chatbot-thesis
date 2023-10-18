from flask import Flask, request, jsonify
from controller.app_controller import AppController
import werkzeug.exceptions as werexc
from flask_cors import CORS
from config import app_config


app = Flask(__name__)
CORS(app)


#Użycie dekoratora app.before_request pozwala za każdym razem
#przed wykonaniem zapytania sprawdzić czy
#adres IP znajduje się w liście dostępnych
@app.before_request
def check_request_client():
    status = AppController.check_client(request.remote_addr)
    if not status[0]:
        return jsonify({"Response": f"{status[1]}"})

#Dwie funkcje odpowiedzialne za zarządzanie Exception w programie
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"EXCEPTION": f"{e}"})


@app.errorhandler(werexc.HTTPException)
def handle_error(e):
    return jsonify({"EXCEPTION": f"{e}"})


#Endpointy oraz ich logika wykonań
@app.get('/')
def index():
    return "Hello"


@app.post('/talk')
def predict():
    data = request.get_json().get('message')
    response = AppController.get_response(data)
    return jsonify({"Response": f"{response}"})


@app.get('/training')
def training():
    AppController.run_training()
    return jsonify({"Message": "SUCCESS"})


#Inicjalizacja aplikacji serwerowej na podstawie wybranych parametrów
if __name__ == '__main__':
    app.run(debug=False, host=app_config.HOST, port=app_config.PORT)
