.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "Usage: make [option]"
	@echo ""
	@echo "Options:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

.PHONY: setup
install:  ## Install dependencies
	pip install -r requirements.txt

.PHONY: data
data:  ## Prepare data for the app (will overwrite any existing data)
	cd dev && python prepare_data.py && cd .. && python dev/move_files_to_app.py

.PHONY: start
start:  ## Start app
	streamlit run app/app.py

.PHONY: del-audio
del-audio:  ## Delete any generated audio files
	rm -f *.mp3
