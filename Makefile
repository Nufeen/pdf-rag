.PHONY: venv install setup clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

PYTHON3 := python3.13

venv:
	@$(PYTHON3) -c "import sys; v=sys.version_info; exit(0 if v>=(3,11) else 1)" || \
	  (echo "Error: Python 3.11+ required. Found: $$($(PYTHON3) --version 2>&1)"; \
	   echo "Install with: brew install python@3.13"; exit 1)
	$(PYTHON3) -m venv $(VENV)
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
