import os
from typing import List
from datetime import datetime, timedelta
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import finnhub as fh


def get_social_sentiment(
    symbol: str, _from: datetime, to: datetime, lookback: timedelta
) -> pd.DataFrame:
    load_dotenv()
    finnhub_api_key = os.getenv("FINNHUB_API_KEY", None)
    fh_client = fh.Client(api_key=finnhub_api_key)

    from_ = (_from - lookback).strftime("%Y-%m-%d")
    to_ = to.strftime("%Y-%m-%d")
    social_sentiment = fh_client.stock_social_sentiment(symbol, _from=from_, to=to_)

    sentiment_data_frames = []
    sentiment_platforms = ["reddit", "twitter"]
    for sentiment_platform in sentiment_platforms:
        sentiment_df = pd.DataFrame(social_sentiment[sentiment_platform])
        if len(sentiment_df) == 0:
            continue
        sentiment_df["atTime"] = (
            pd.to_datetime(sentiment_df["atTime"]).astype(np.int64) // 10**9
        )
        sentiment_df = sentiment_df.rename(
            columns={
                "atTime": "time",
                "positiveScore": "postitive_score",
                "negativeScore": "negative_score",
                "negativeMention": "negative_mention",
                "positiveMention": "positive_mention",
            }
        )
        sentiment_df = sentiment_df.rename(
            columns={
                column: f"{sentiment_platform}_{column}"
                for column in sentiment_df.columns
                if column != "time"
            }
        )
        sentiment_data_frames.append(sentiment_df)

    if len(sentiment_data_frames) == 2:
        sentiment_df = sentiment_data_frames[0].merge(
            sentiment_data_frames[1], how="outer"
        )
        sentiment_df = sentiment_df.fillna(value=0.0)
        sentiment_df = sentiment_df.sort_values(["time"])
    else:
        sentiment_df = pd.DataFrame()
    return sentiment_df


def get_social_sentiment_alternatives(
    symbol: str, days: List[int], lookback: timedelta
) -> pd.DataFrame:
    load_dotenv()
    finnhub_api_key = os.getenv("FINNHUB_API_KEY", None)
    fh_client = fh.Client(api_key=finnhub_api_key)

    # from_ = (_from - lookback).strftime("%Y-%m-%d")
    # to_ = to.strftime("%Y-%m-%d")
    all_sentiments = []
    for day in days:
        # from_ = datetime.fromtimestamp(day).strftime("%Y-%m-%d")
        social_sentiment = fh_client.stock_social_sentiment(symbol, _from=day)

        sentiment_data_frames = []
        sentiment_platforms = ["reddit", "twitter"]
        for sentiment_platform in sentiment_platforms:
            sentiment_df = pd.DataFrame(social_sentiment[sentiment_platform])
            if len(sentiment_df) == 0:
                continue
            sentiment_df["atTime"] = (
                pd.to_datetime(sentiment_df["atTime"]).astype(np.int64) // 10**9
            )
            sentiment_df = sentiment_df.rename(
                columns={
                    "atTime": "time",
                    "positiveScore": "postitive_score",
                    "negativeScore": "negative_score",
                    "negativeMention": "negative_mention",
                    "positiveMention": "positive_mention",
                }
            )
            sentiment_df = sentiment_df.rename(
                columns={
                    column: f"{sentiment_platform}_{column}"
                    for column in sentiment_df.columns
                    if column != "time"
                }
            )
            sentiment_data_frames.append(sentiment_df)

        if len(sentiment_data_frames) == 2:
            sentiment_df = sentiment_data_frames[0].merge(
                sentiment_data_frames[1], how="outer"
            )
            sentiment_df = sentiment_df.fillna(value=0.0)
            all_sentiments.append(sentiment_df)

    sentiments_df = pd.concat(all_sentiments)
    return sentiments_df


if __name__ == "__main__":
    get_social_sentiment("AAPL", 1660892400, 1660892800)
