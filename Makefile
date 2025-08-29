.PHONY: init
init:
	cp .env.example .env
	poetry install --extras "all" --no-root

.PHONY: install
install: 
	poetry install --no-root

.PHONY: dev
dev:
	poetry run uvicorn src.main:app --reload

.PHONY: test
test:
	poetry run python -m pytest

.PHONY: lint
lint:
	poetry run pylint .

.PHONY: check
check:
	poetry run pyright .

.PHONY: format
format:
	poetry run black .