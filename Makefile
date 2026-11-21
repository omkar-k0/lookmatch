.PHONY: check lint test spike frames eval

check: lint test   ## what CI runs

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

spike:             ## rerun the 17 Nov spike on the package code
	uv run python -m lookmatch.spike

frames:            ## data/footage/clips -> data/frames (sampled, resized frames)
	uv run python -m lookmatch.sample data/footage

eval:              ## dev split by default; SPLIT=heldout only once per milestone
	uv run python -m lookmatch.evaluate --split $(or $(SPLIT),dev)
