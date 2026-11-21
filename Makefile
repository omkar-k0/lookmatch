.PHONY: check lint test spike footage frames eval

check: lint test   ## what CI runs

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

spike:             ## rerun the 17 Nov spike on the package code
	uv run python -m lookmatch.spike

footage:           ## download the exact eval clips listed in eval/footage_manifest.csv (needs internet)
	python3 scripts/fetch_footage.py --from-manifest eval/footage_manifest.csv

frames:            ## data/footage/clips -> data/frames (sampled, resized frames)
	uv run python -m lookmatch.sample data/footage

eval:              ## dev split by default; SPLIT=heldout only once per milestone
	uv run python -m lookmatch.evaluate --split $(or $(SPLIT),dev)
	uv run python -m lookmatch.report
