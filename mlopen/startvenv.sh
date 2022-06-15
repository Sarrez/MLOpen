#!/usr/bin/env python3

echo 'Starting venv from script and executing pipeline...'
python3 -m pip install virtualenv
. $1/bin/activate
echo 'running venv'
python -c 'import runpipe; runpipe.run()' $2 $3 $4
pip -V