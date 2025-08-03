help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)
	@echo ''

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

run_app: ## Hosts streamlit chat interface
	cd src && python3 -m streamlit run app/Chat_Text2ExtendedCypher.py

run_app_hl: ## Hosts streamlit chat interface headless, does not start new browser window
	cd src && python3 -m streamlit run app/Chat_Text2ExtendedCypher.py --server.headless true
