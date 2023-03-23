#!/bin/sh

SCRIPT_DIR=$(dirname $(dirname $0))

source $SCRIPT_DIR/.venv/bin/activate
python $SCRIPT_DIR/console.py train