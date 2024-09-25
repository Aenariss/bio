# BIO Project 2024 Makefile to make the setup simpler
# Author: Vojtech Fiala <xfiala61> + ChatGPT

PYTHON := $(shell command -v python3 || command -v python)

run:
	$(PYTHON) main.py

test:
	$(PYTHON) test.py

build:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
