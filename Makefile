VENV := .venv

ifeq ($(OS),Windows_NT)
   BIN=$(VENV)/Scripts
else
   BIN=$(VENV)/bin
endif

export PATH := $(BIN):$(PATH)

PROJECT := service
TESTS := tests

IMAGE_NAME := reco_service
CONTAINER_NAME := reco_service

# Prepare

.venv:
	poetry install --no-root
	poetry check

setup: .venv
	pip install implicit==0.4.4 lightfm==1.16

script:
	./config/load_light_fm.sh

# Clean

clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf $(VENV)


# Format

isort_fix: .venv
	isort $(PROJECT) $(TESTS)

blake: .venv
	black $(PROJECT) $(TESTS) -l 79

format: isort_fix blake


# Lint

isort: .venv
	isort --check $(PROJECT) $(TESTS)

flake: .venv
	flake8 $(PROJECT) $(TESTS)

mypy: .venv
	mypy $(PROJECT) $(TESTS)

pylint: .venv
	pylint $(PROJECT) $(TESTS)

lint: isort flake mypy pylint


# Test

.pytest:
	pytest

test: .venv .pytest


# Docker

build:
	docker build . -t $(IMAGE_NAME)

run: build
	docker run -p 8080:8080 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# All

all: setup format lint test run

.DEFAULT_GOAL = all