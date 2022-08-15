import os
from lemon import api
from dotenv import load_dotenv


def create_api_client():
    load_dotenv()
    client = api.create(
        market_data_api_token=os.getenv("MARKET_DATA_API_TOKEN", None),
        trading_api_token=os.getenv("TRADING_API_TOKEN", None),
        env="paper",
    )
    return client
