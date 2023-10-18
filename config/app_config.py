#Plik konfiguracyjny zawierający wymagane do uruchomienia programu parametry

#Sekcja ścieżek modelu 
INTENTS_PATH = r"C:\Users\SKASUS\Desktop\Chatbot\intents\intents.json"
MODEL_DATA_PATH = r"C:\Users\SKASUS\Desktop\Chatbot\ai_files\model_data.pth"

#Whitelista adresów IP które otrzymują dostęp do aplikacji
IP_WHITELIST = ["127.0.0.1", ""]

#Sekcja dostępu do API
API_KEY = "0675fed462b94a3692d95914232003"
API_DEFAULT_URL = "http://api.weatherapi.com/v1"

#Sekcja dostępu do bazy danych
DB_PATH = r"C:\Users\SKASUS\Desktop\Chatbot\db\sample_db.db"
DB_USER = ""
DB_PWD = ""

#Sekcja ustawień startowych serwera aplikacji
HOST = "0.0.0.0"
PORT = "55555"