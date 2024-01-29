# Define the default target
.PHONY: run-pipeline
run-pipeline: preprocess train predict validate

# Define variables for paths
DATA_DIR=./data
MODEL_DIR=./models
SRC_DIR=./src
CONFIG=$(SRC_DIR)/config.yaml

# Define variables for data files
RAW_DATA=$(DATA_DIR)/default_of_credit_card_clients.csv
PROCESSED_DATA=$(DATA_DIR)/processed_data
PREDICT_DATA=$(DATA_DIR)/predict_data.csv
MODEL_PATH=$(MODEL_DIR)/output_model.pkl

# Define the command to run Python scripts
POETRY_RUN=poetry run python

# Targets for pipeline stages
.PHONY: preprocess train predict validate clean

preprocess:
	@echo "Preprocessing data..."
	$(POETRY_RUN) $(SRC_DIR)/pipeline.py --config-path $(CONFIG) --stage preprocess

train:
	@echo "Training model..."
	$(POETRY_RUN) $(SRC_DIR)/pipeline.py --config-path $(CONFIG) --stage train

predict:
	@echo "Making predictions..."
	$(POETRY_RUN) $(SRC_DIR)/pipeline.py --config-path $(CONFIG) --stage predict

validate:
	@echo "Validating model..."
	$(POETRY_RUN) $(SRC_DIR)/pipeline.py --config-path $(CONFIG) --stage validate

clean:
	@echo "Cleaning up..."
	rm -f $(PROCESSED_DATA)/*.csv
	rm -f $(MODEL_PATH)

# Target to install Poetry
.PHONY: install-poetry
install-poetry:
	@echo "Installing Poetry..."
	curl -sSL https://install.python-poetry.org | python3 -
	echo "Poetry installed. Please restart your terminal session." 
# Target to install dependencies using Poetry
.PHONY: install-deps
install-deps:
	@echo "Installing dependencies..."
	poetry install