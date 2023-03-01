#!/bin/sh

SCRIPT_DIR=$(dirname $0)

virtualenv --python=3.10 $SCRIPT_DIR/.venv
source $SCRIPT_DIR/.venv/bin/activate
pip install -r $SCRIPT_DIR/requirements.txt