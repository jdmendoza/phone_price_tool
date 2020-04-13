import requests
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import (StringField, SubmitField)
from flask_bootstrap import Bootstrap

import dash
import dash_html_components as html

from flask_nav import Nav
from flask_nav.elements import Navbar, View


API_URL = "https://uoa1yghdj0.execute-api.us-east-2.amazonaws.com/default/price_predict"
SWAPPA_URL = "https://swappa.com/mobile/buy/"

server = Flask(__name__)
server.config['SECRET_KEY'] = 'supersecret'

Bootstrap(server)
nav = Nav()


@nav.navigation()
def mynavbar():
    return Navbar(
        'Phone Price Tool',
        View('Home', 'index'),
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
            url = SWAPPA_URL + phone['model'] + '/' + phone['carrier']

        else:
            error = True

    return render_template('index.html', form=form, price=price, error=error, url=url)


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/admin/'
)

app.layout = html.Div("My Dash app")
nav.init_app(server)


if __name__ == "__main__":
    app.run_server(debug=True)
