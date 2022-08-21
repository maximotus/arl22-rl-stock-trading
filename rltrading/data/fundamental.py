# import requests
# from bs4 import BeautifulSoup
# import pandas as pd


# base_path = "https://traderfox.de/aktien/{0}/fundamental"

# print(base_path.format('40678-apple-inc'))
# page = requests.get(base_path)
# soup = BeautifulSoup(page.text, 'lxml')
# print(soup)

# table = soup.find('table')
# print(table)


from http import client
import finnhub as fh


finnhub_client = fh.Client(api_key="cc0uvnaad3ifk6takpvg")
print(finnhub_client.stock_social_sentiment('GME'))
