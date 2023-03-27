#!/bin/sh

SCRIPT_DIR=$(dirname $(dirname $0))

rm -rf $SCRIPT_DIR/.venv
python3 -m virtualenv --python=3.10 $SCRIPT_DIR/.venv
source $SCRIPT_DIR/.venv/bin/activate
pip install -r $SCRIPT_DIR/requirements.txt