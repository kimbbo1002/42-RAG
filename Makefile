PYTHON = uv run python3
SRC_DIR = src


install:
	uv sync

run:
	$(PYTHON) -m $(SRC_DIR) search "How to configure OpenAI server?" --k 5

debug:
	$(PYTHON) -m pdb -m $(SRC_DIR) search "How to configure OpenAI server?" --k 5

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	find . -name "*.pyc" -delete

fclean: clean
	rm -rf uv.lock
	rm -rf .venv

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean fclean lint