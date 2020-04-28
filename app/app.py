import os
import requests
import datetime
from collections import OrderedDict
import boto3
import pandas as pd

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import (StringField, SubmitField)
from flask_bootstrap import Bootstrap

from flask_nav import Nav
from flask_nav.elements import Navbar, View, Link

import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
import dash

API_URL = "https://uoa1yghdj0.execute-api.us-east-2.amazonaws.com/default/price_predict"
SWAPPA_URL = "https://swappa.com/mobile/buy/"

model_map = {'iPhone 6s': 'apple-iphone-6s', 'iPhone 6s Plus': 'apple-iphone-6s-plus', 'iPhone 7': 'apple-iphone-7',
             'iPhone 8': 'apple-iphone-8', 'iPhone 8 Plus': 'apple-iphone-8-plus', 'iPhone X': 'apple-iphone-x',
             'iPhone XR': 'apple-iphone-xr', 'iPhone XS': 'apple-iphone-xs', 'iPhone XS Max': 'apple-iphone-xs-max',
             'iPhone 11': 'apple-iphone-11'}

server = Flask(__name__)
server.config['SECRET_KEY'] = 'supersecret'

Bootstrap(server)
nav = Nav()


def to_datetime(dt_str):
    return datetime.datetime.strptime(dt_str, '%Y%m%d%H%M%S')


@nav.navigation()
def mynavbar():
    return Navbar(
        'Phone Price Tool',
        View('Home', 'index'),
        Link('Admin', 'https://phone-price-tool.herokuapp.com/admin'),
    )


class InfoForm(FlaskForm):

    model = StringField(" ")
    storage = StringField(" ")
    condition = StringField(" ")
    carrier = StringField(" ")
    color = StringField(" ")

    submit = SubmitField(" ")


@server.route('/', methods=['GET', 'POST'])
def index():
    form = InfoForm()
    price = None
    error = None
    url = None
    phone = None
    phone_details = None

    if request.method == 'POST':
        phone = OrderedDict()

        phone['model'] = form.model.data
        phone['condition'] = form.condition.data
        phone['storage'] = form.storage.data
        phone['carrier'] = form.carrier.data
        phone['color'] = form.color.data

        if all(phone.values()):
            phone_details = ' - '.join(phone.values())

            phone['model'] = model_map[phone['model']]
            phone['storage'] = int(phone['storage'])
            error = None
            result = requests.post(API_URL, json=phone)
            price = result.text
            url = SWAPPA_URL + phone['model'] + '/' + phone['carrier']

        else:
            error = True

    return render_template('index.html', form=form, price=price, error=error, url=url, phone_details=phone_details)


def mse_vis():
    session = boto3.Session(region_name='us-east-2',
                           aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                           aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

    metrics_table = session.resource('dynamodb').Table('model_metrics')
    response = metrics_table.scan()
    data = [(to_datetime(x['id']), x['metric'], float(x['value'])) for x in response['Items']]
    df = pd.DataFrame(data, columns=['datetime', 'metric', 'value']).sort_values(by='datetime', ascending=True)

    return html.Div([
        dcc.Graph(id='mse-plot',
                  config={'displayModeBar': True},
                  figure={'data': [
                        go.Scatter(x=df['datetime'], y=df['value'], mode='lines+markers')
                            ]})
        ])


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/admin/'
)

app.layout = html.Div([dcc.Markdown('''## Model Performance Dashboard '''),
                       dcc.Markdown(''' This dashboard tracks the MSE for the model when it is updated.'''),
                       mse_vis()], style={'font': 'helvetica-neue', 'color': '#343A40'})
nav.init_app(server)


if __name__ == "__main__":
    app.run_server()
