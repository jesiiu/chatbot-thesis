from ai_files.logic import ChatbotLogic
from ai_files.training import ChatbotTraining as Training
import config.app_config as app_config

#Kontroler aplikacji przekierowujący żądanie w odpowiednie miejsce
class AppController:

    #Funckja sprawdzająca czy IP z którego zostało wysłane zapytanie znajduje się na White-List
    @staticmethod
    def check_client(data):
        if data not in app_config.IP_WHITELIST:
            return (False, "You have no accesss to use this Chatbot")
        else:
            return (True, "Access granted")

    #Funkcja przekazująca intencje użytkownika do modelu i zwracająca odpowiedź
    @staticmethod
    def get_response(message):
        chat_bot = ChatbotLogic()
        response = chat_bot.get_answer(message)
        return response

    #Funckja wywołująca trening modelu
    @staticmethod
    def run_training():
        bot_train = Training()
        bot_train.train()
