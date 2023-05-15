#! /bin/sh -e

run() {
    python3 -m venv .
    source ../scripts/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r ../requirements.txt
}