#!/bin/sh

SCRIPT_DIR=$(dirname $0)

source $SCRIPT_DIR/.venv/bin/activate
python $SCRIPT_DIR/console.py predict $1