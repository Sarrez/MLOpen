#!/usr/bin/env python3

echo 'Starting venv from script and executing pipeline...'
python3 -m pip install virtualenv
. $1/bin/activate
python3 -m pip install plotly
python3 -m pip uninstall utils
python3 -m pip install Pillow
python3 -m pip install sqlalchemy
python3 -m pip install torch
echo 'running venv'
python -c 'import runpipe; runpipe.run()' $2 $3 $4
pip -V