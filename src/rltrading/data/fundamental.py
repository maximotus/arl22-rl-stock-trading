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

import pandas as pd
import finnhub as fh


finnhub_client = fh.Client(api_key="cc0uvnaad3ifk6takpvg")
print(finnhub_client.stock_social_sentiment("GME"))


def get_social_sentiment(symbol: str, fh_client: fh.Client) -> pd.DataFrame:
    pass


if __name__ == "__main__":
    pass
