# from time import sleep, time
# from tkinter import E
# from typing import List
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
#
# import numpy as np
# import pandas as pd
# import finnhub as fh
# from finnhub.exceptions import FinnhubAPIException
#
#
# def get_social_sentiment(
#     api_key: str, symbol: str, _from: datetime, to: datetime, lookback: timedelta
# ) -> pd.DataFrame:
#     """Fetches social sentiment scores for ``reddit`` and ``twitter``
#     in a given timeframe.
#
#     The timeframe has a lower bound of: ``_from - lookback`` and an
#     upper bound of ``to``.
#
#     Parameters
#     ----------
#     api_key: str
#         The API key used to access the ``Finnhub`` database.
#     symbol : str
#         The symbol of the asset to fetch the data for.
#     _from : datetime
#         From when the data should be fetched for.
#     to : datetime
#         To when the data should be fetched for.
#     lookback : timedelta
#         An additional lookback.
#
#     Returns
#     -------
#     pd.DataFrame
#         A dataframe containing the social sentiment
#         data for ``reddit`` and ``twitter``.
#     """
#     fh_client = fh.Client(api_key=api_key)
#
#     from_ = (_from - lookback).strftime("%Y-%m-%d")
#     to_ = to.strftime("%Y-%m-%d")
#     social_sentiment = fh_client.stock_social_sentiment(symbol, _from=from_, to=to_)
#
#     sentiment_data_frames = []
#     sentiment_platforms = ["reddit", "twitter"]
#     for sentiment_platform in sentiment_platforms:
#         sentiment_df = pd.DataFrame(social_sentiment[sentiment_platform])
#         if len(sentiment_df) == 0:
#             continue
#         sentiment_df["atTime"] = (
#             pd.to_datetime(sentiment_df["atTime"]).astype(np.int64) // 10**9
#         )
#         sentiment_df = sentiment_df.rename(
#             columns={
#                 "atTime": "time",
#                 "positiveScore": "postitive_score",
#                 "negativeScore": "negative_score",
#                 "negativeMention": "negative_mention",
#                 "positiveMention": "positive_mention",
#             }
#         )
#         sentiment_df = sentiment_df.rename(
#             columns={
#                 column: f"{sentiment_platform}_{column}"
#                 for column in sentiment_df.columns
#                 if column != "time"
#             }
#         )
#         sentiment_data_frames.append(sentiment_df)
#
#     if len(sentiment_data_frames) == 2:
#         sentiment_df = sentiment_data_frames[0].merge(
#             sentiment_data_frames[1], how="outer"
#         )
#         sentiment_df = sentiment_df.fillna(value=0.0)
#         sentiment_df = sentiment_df.sort_values(["time"])
#     else:
#         sentiment_df = pd.DataFrame(columns=["time"])
#     return sentiment_df
#
#
# def get_social_sentiment_alternative(
#     finnhub_api_key: str, symbol: str, days: List[int], lookback: timedelta
# ) -> pd.DataFrame:
#     fh_client = fh.Client(api_key=finnhub_api_key)
#
#     all_sentiments = []
#     for day in days:
#         from_ = (day - lookback).strftime("%Y-%m-%d")
#         to_ = day.strftime("%Y-%m-%d")
#
#         social_sentiment = __get_social_sentiment_df(fh_client, symbol, from_, to_)
#
#         sentiment_data_frames = []
#         sentiment_platforms = ["reddit", "twitter"]
#         for sentiment_platform in sentiment_platforms:
#             sentiment_df = pd.DataFrame(social_sentiment[sentiment_platform])
#             if len(sentiment_df) == 0:
#                 continue
#             sentiment_df["atTime"] = (
#                 pd.to_datetime(sentiment_df["atTime"]).astype(np.int64) // 10**9
#             )
#             sentiment_df = sentiment_df.rename(
#                 columns={
#                     "atTime": "time",
#                     "positiveScore": "postitive_score",
#                     "negativeScore": "negative_score",
#                     "negativeMention": "negative_mention",
#                     "positiveMention": "positive_mention",
#                 }
#             )
#             sentiment_df = sentiment_df.rename(
#                 columns={
#                     column: f"{sentiment_platform}_{column}"
#                     for column in sentiment_df.columns
#                     if column != "time"
#                 }
#             )
#             sentiment_data_frames.append(sentiment_df)
#
#         if len(sentiment_data_frames) == 2:
#             sentiment_df = sentiment_data_frames[0].merge(
#                 sentiment_data_frames[1], how="outer"
#             )
#             sentiment_df = sentiment_df.fillna(value=0.0)
#             print(len(sentiment_df))
#             all_sentiments.append(sentiment_df)
#
#     sentiments_df = pd.concat(all_sentiments)
#     return sentiments_df
#
#
# def __get_social_sentiment_df(fh_client, symbol, from_, to_):
#     try:
#         return fh_client.stock_social_sentiment(symbol, _from=from_, to=to_)
#     except FinnhubAPIException:
#         sleep(2.0)
#         return __get_social_sentiment_df(fh_client, symbol, from_, to_)
