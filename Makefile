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
# 	теперь implicit и lightfm загружаются через poetry
#	pip install implicit==0.4.4 lightfm==1.16

load_models:
	# запустить make load_models (бывший make script) перед запуском сервера или сборкой докера
	./load_models_from_google_drive.sh

# Clean

clean:
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf $(VENV)


# Format

isort_fix: .venv
	isort $(PROJECT) $(TESTS)

black: .venv
	black $(PROJECT) $(TESTS) -l 88

format: isort_fix black


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
	docker run -p 80:80 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# All

all: setup format lint test run

.DEFAULT_GOAL = all
