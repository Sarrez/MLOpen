#!/bin/bash

echo 'Executing script..'
python3 -m venv $1
. $1/bin/activate
python3 -m pip install -r $2
echo 'installed reqs'
deactivate
pip -V