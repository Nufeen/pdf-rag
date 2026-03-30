.PHONY: venv install setup clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)
	@echo ""
	@echo "Activate with:"
	@echo "  source $(VENV)/bin/activate"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .

setup: install
	@echo ""
	@echo "Setup complete. To get started:"
	@echo "  source $(VENV)/bin/activate"
	@echo "  cp .env.example .env   # then edit .env with your Ollama host IP"
	@echo "  pdf-rag index ~/Books/"

clean:
	rm -rf $(VENV) *.egg-info dist build __pycache__ pdf_rag/__pycache__
