#!/bin/bash

echo 'Executing script..'
python3 -m venv $1
. $1/bin/activate
python3 -m pip install -r $2
python3 -m pip install django
python3 -m pip install matplotlib
python3 -m pip install plotly
python3 -m pip install Pillow
python3 -m pip install sqlalchemy
python3 -m pip install django-bootstrap4
python3 -m pip install psycopg2
python3 -m pip install torch
echo 'installed reqs'
deactivate
