.PHONY: init
init:
	cp .env.example .env
	poetry install --extras "all" --no-root

.PHONY: install
install: 
	poetry install --no-root

.PHONY: docker-dev
docker-dev:
	docker compose up

.PHONY: dev
dev:
	poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: worker
worker:
	poetry run celery -A src.celery.celery_app worker --loglevel=info --concurrency=2

.PHONY: redis-start
redis-start:
	redis-server --daemonize yes --port 6379

.PHONY: redis-stop
redis-stop:
	redis-cli shutdown

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