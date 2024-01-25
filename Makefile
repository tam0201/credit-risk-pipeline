.PHONY: create-fraud-detection-env install-deps test clean all

# Virtual environment setup
create-fraud-detection-env:
	@if [ ! -d "fraud-detection" ]; then \
		echo "Creating virtual environment for fraud detection..."; \
		python3 -m venv fraud-detection; \
		echo "Virtual environment for fraud detection created."; \
	else \
		echo "Virtual environment for fraud detection already exists."; \
	fi

# Dependency installation
install-deps: create-fraud-detection-env
	@echo "Installing dependencies..."
	@. fraud-detection/bin/activate; pip install -r requirements.txt; mim install mmengine
	@echo "Dependencies installed."

# Run tests
test: install-deps
	@echo "Running tests..."
	@. fraud-detection/bin/activate; pytest
	@echo "Tests completed."

# Clean the environment
clean:
	@echo "Cleaning up..."
	@rm -rf fraud-detection
	@echo "Cleaned the virtual environment for fraud detection."

# All-in-one command
all: create-fraud-detection-env install-deps test