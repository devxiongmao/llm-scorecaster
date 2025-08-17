.PHONY: init
init:
	cp .env.example env
	poetry install --no-root

.PHONY: install
install: 
	poetry install --no-root

.PHONY: dev
dev:
	poetry run uvicorn src.main:app --reload