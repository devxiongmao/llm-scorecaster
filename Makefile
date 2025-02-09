.PHONY: install
install: 
	poetry install --no-root

.PHONY: determine-version
determine-version:
	poetry run python util/determine_version.py

.PHONY: generate-changelog
generate-changelog:
	poetry run python util/generate_changelog.py
