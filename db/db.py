import sqlite3
from config import app_config


class Db:
    #Inicjalizacja połączenia z bazą danych
    def __init__(self):
        self.client = sqlite3.connect(app_config.DB_PATH)
        # self.init = self.__initialize_database()

    #Poniżej znajdują się funkcje odpowiedzialne za wykonywanie zapytań do bazy danych
    def execute_query_tuple(self, query, tuple):
        cursor = self.client.cursor()
        cursor.execute(query, tuple)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def execute_scalar_tuple(self, query, tuple):
        cursor = self.client.cursor()
        cursor.execute(query, tuple)
        rows = cursor.fetchone()[0]
        cursor.close()
        return rows

    def insert_into_tuple(self, query, tuple):
        cursor = self.client.cursor()
        cursor.execute(query, tuple)
        self.client.commit()

    #Przygotowane funkcje inicjalizujące podstawowe dane w bazie danych
    def __initialize_database(self):
        cursor = self.client.cursor()
        cursor.execute('''CREATE TABLE data (question text, answer text)''')
        self.client.commit()

    def __insert_samples(self):
        cursor = self.client.cursor()
        cursor.execute(
            '''INSERT INTO orders VALUES (1, 'new'), (2, 'in progres'), (3, 'finalized')''')
        self.client.commit()
