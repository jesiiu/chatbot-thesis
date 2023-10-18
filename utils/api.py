import requests
import json
import config.app_config as app_config

#Klasa Api odpowiedzialna za przetwarzanie zapytań wysyłanych przez
#wybranie odpowiedniej intencji
class Api:
    def __init__(self) -> None:
        self.token = app_config.API_KEY
        self.base_url = app_config.API_DEFAULT_URL

    def make_request(self, method, endpoint, data=None, headers=None, params=None):
        """Method for send request to server

        Args:
            method (str): Type of API method ex. 'GET', 'POST'
            endpoint (str): Name of endpoint that you want to connect ex. '/login'
            data (json): Provide data that you want to send to API. Defaults to None.
            headers (_type_, optional): Headers for request. Defaults to None.
            params (_type_, optional): Params for request. Defaults to None.

        Raises:
            Exception: If status is different that 200 raise Exception

        Returns:
            response (json): Return from API
        """
        if data is None:
            data = json.dumps({})

        url = self.base_url + endpoint

        # params = json.dumps({"key": self.token, "q": f"{params}"})
        params = {"key": self.token, "q": f"{params}"}

        response = requests.request(
            method=method,
            url=url,
            data=data,
            headers=headers,
            params=params
        )

        if response.status_code in (200, 201):
            return response.json()
        else:
            raise Exception(response.content)
