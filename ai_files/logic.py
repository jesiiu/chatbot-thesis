import random
import json
import torch
from ai_files.model import NeuralNet
from utils.language_utils import Utils
from config import app_config
from utils.api import Api
from db.db import Db


class ChatbotLogic:
    #Inicjalizacja modelu który jest odpowiedzialny za rozpoznanie intecji
    #oraz udzielenie odpowiedzi 
    def __init__(self):
        self.intents = self.load_intents()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.data = torch.load(app_config.MODEL_DATA_PATH)
        self.name = "Eva"
        self.input_size = self.data["input_size"]
        self.hidden_size = self.data["hidden_size"]
        self.output_size = self.data["output_size"]
        self.words_list = self.data['words_list']
        self.tags = self.data['tags']
        self.model_state = self.data["model_state"]

        self.model = NeuralNet(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_classes=self.output_size).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

        self.utils = Utils()

    #Funkcja ładująca plik intencji oraz odpowiedzi do pamięci
    def load_intents(self):
        with open(app_config.INTENTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    #Funckja przetwarzająca wiadomość użytkownika oraz zwracająca odpowiednią odpowiedź
    def get_answer(self, message):
        sentence = self.utils.tokenize_word(message)
        x = self.utils.bag_of_words(sentence, self.words_list)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(self.device)

        output = self.model(x)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        propably = torch.softmax(output, dim=1)
        p = propably[0][predicted.item()]
        if p.item() > 0.85:
            for intent in self.intents['intents']:
                #Blok odpowiedzialny za przetwarzanie informacji intencji pogodowej
                if tag == 'pogoda':
                    weather_intent = [
                        x for x in self.intents['intents'] if tag == x['tag']]
                    temp_data = self.__request_for_weather(
                        message, weather_intent[0])
                    self.__collect_data(message, temp_data)
                    return temp_data

                #Blok odpowiedzialny za przetwarzanie informacji intencji statusu zamówienia
                if tag == 'status_zamowienia':
                    order_intent = [
                        x for x in self.intents['intents'] if tag == x['tag']]
                    temp_data = self.__request_for_db(message, order_intent[0])
                    self.__collect_data(message, temp_data)
                    return temp_data

                #Blok odpowiedzialny za przetwarzanie pozostałych intencji
                if tag == intent['tag']:
                    temp_data = random.choice(intent['responses'])
                    self.__collect_data(message, temp_data)
                    return temp_data
                else:
                    return "Wybacz ale nie zrozumiałem Twojego pytania, czy mógłbyś powtórzyć?"

    #Funkcja wysyłająca żądanie do API w celu otrzymania aktualnej pogody w danym mieście
    def __request_for_weather(self, sentence, intent):
        try:
            client = Api()
            city = self.utils.get_city_name(sentence)
            if city is None or (city[0] is None and city[1] is None):
                return """Wybacz ale nie jestem wstanie rozpoznać wprowadzonej przez Ciebie nazwy miejscowości, 
            sprawdź pisownię i spróbuj ponownie, najlepiej z wielkiej litery"""

            response = client.make_request(
                method='GET', endpoint='/current.json', params=city[0])
            chatbot_response = random.choice(intent['responses'])
            return chatbot_response.format(city[1], response['current']['temp_c'])

        except:
            return f"""Wybacz ale nie mogę znaleść miasta {city[1]} na mojej mapie, 
        sprawdź poprawność pisowni ewentualnie podaj nazwę miasta w języku angielskim 
        (jeżeli jest taka możliwość)"""

    #Funkcja wysyłająca zapytanie do bazy danych w celu otrzymania informacji na temat danego zamówienia
    def __request_for_db(self, sentence, intent):
        try:
            db = Db()
            order_id = self.utils.get_order_id(sentence)
            order_status = db.execute_scalar_tuple(
                '''SELECT status FROM orders WHERE order_id = ?''', (order_id,))
            chatbot_response = random.choice(intent['responses'])
            return chatbot_response.format(order_id, order_status)
        except:
            return F"""Niestety ale nie mogę znaleźć zamówienia o podanym przez Ciebie numerze {order_id}, sprawdź czy został wprowadzony prawidłowo i spróbuj ponownie."""

    #Funkcja odpowiedzialna za zapisywanie wysłanych zapytań oraz odpowiedzi do bazy danych
    def __collect_data(self, question, answer):
        try:
            db = Db()
            db.insert_into_tuple(
                "INSERT INTO data (question, answer) values (?, ?)", (question, answer))
        except:
            return f"""Db error"""
