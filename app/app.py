import requests
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import (StringField, SubmitField)
from flask_bootstrap import Bootstrap

import dash
import dash_html_components as html

API_URL = "https://uoa1yghdj0.execute-api.us-east-2.amazonaws.com/default/price_predict"

server = Flask(__name__)
server.config['SECRET_KEY'] = 'supersecret'

Bootstrap(server)


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

    if request.method == 'POST':
        phone = dict()

        phone['model'] = form.model.data
        phone['condition'] = form.condition.data
        phone['storage'] = form.storage.data
        phone['carrier'] = form.carrier.data
        phone['color'] = form.color.data

        if all(phone.values()):
            phone['storage'] = int(phone['storage'])
            error = None
            result = requests.post(API_URL, json=phone)
            price = result.text

        else:
            error = True

    return render_template('index.html', form=form, price=price, error=error)


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/admin/'
)

app.layout = html.Div("My Dash app")


if __name__ == "__main__":
    app.run_server(debug=True)
