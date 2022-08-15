import json

import paho.mqtt.client as mqtt

from candlestick_chart import Candle, Chart

from rltrading.data.utils import create_api_client


# PLOTLY
chart = Chart(candles=[], title="Test")
chart.set_name("Test/Test")
chart.draw()

# SUBSCRIPTION


class Quote:
    def __init__(self, data):
        self.isin = data["isin"]
        self.mic = data["mic"]
        self.bid_volume = data["b_v"]
        self.ask_volume = data["a_v"]
        self.ask = data["a"]
        self.bid = data["b"]
        self.time = data["t"]


client = create_api_client()
token = client.streaming.authenticate()


def on_connect(token):
    def _on_connect(mqtt_client, userdata, flags, rc):
        mqtt_client.subscribe(token.user_id)

    return _on_connect


def on_subscribe(token, isins):
    def _on_sub(mqtt_client, userdata, level, buff):
        mqtt_client.publish(f"{token.user_id}.subscriptions", isins)

    return _on_sub


def on_message():
    def _on_msg(mqtt_client, userdata, msg):
        data = json.loads(msg.payload)
        quote = Quote(data)
        # print(quote)
        # open", "close", "high", "low
        high = quote.ask if quote.ask > quote.bid else quote.bid
        low = quote.ask if quote.ask < quote.bid else quote.bid
        open = quote.ask if quote.ask > quote.bid else quote.bid
        close = quote.ask if quote.ask < quote.bid else quote.bid
        chart.update_candles([Candle(open=open, close=close, high=high, low=low)])
        chart.draw()

    return _on_msg


mqttc = mqtt.Client("Ably_Client")
mqttc.username_pw_set(username=token.token)
mqttc.on_subscribe = on_subscribe(token, "DE0007100000")
mqttc.on_connect = on_connect(token)
mqttc.on_message = on_message()

mqttc.connect("mqtt.ably.io")
mqttc.loop_forever()
