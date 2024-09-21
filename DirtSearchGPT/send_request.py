# send_request.py

from ipaddress import ip_address
import re
import requests
import json
from pprint import pprint
import urllib.parse

def send_post_request(api_url, input_text):
    """
    Sends a POST request to the specified API URL with the given input text.

    :param api_url: The endpoint URL of the FastAPI application.
    :param input_text: The text input to be processed by the API.
    :return: The API's response as a JSON object.
    """
    # Define the payload
    payload = {
        "text": input_text
    }

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send the POST request
        response = requests.post(api_url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse and return the JSON response
            return response.json()
        else:
            # Handle unsuccessful responses
            print(f"Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle exceptions (e.g., network errors)
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Replace with your FastAPI endpoint URL
    api_endpoint = "http://127.0.0.1:8000/process-text"

    # Replace with your desired input text
    #input_text = "Singapore does not need visa for indian tourists"
    #input_text = "Donald Trump is a supporter of LGBTQ+ rights"
    input_text = "Donald Trump is not a supporter of LGBTQ+ rights"
    
    #input_text = "Removing of wool from sheep"
    # Send the POST request
    result = send_post_request(api_endpoint, input_text)

    pprint(result)
