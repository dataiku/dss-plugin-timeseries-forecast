# Makefile variables set automatically
plugin_id=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['id']).replace('/',''))"`
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`


plugin:
	@echo "[START] Archiving plugin to dist/ folder..."
	@cat plugin.json | json_pp > /dev/null
	@rm -rf dist
	@mkdir dist
	@echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json
	@git archive -v -9 --format zip -o dist/${archive_file_name} HEAD
	@zip -u dist/${archive_file_name} release_info.json
	@rm release_info.json
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

unit-tests:
	@echo "[START] Running unit tests..."
	@( \
		PYTHON_VERSION=`python3 -V 2>&1 | sed 's/[^0-9]*//g' | cut -c 1,2`; \
		PYTHON_VERSION_IS_CORRECT=`cat code-env/python/desc.json | python3 -c "import sys, json; print(str($$PYTHON_VERSION) in [x[-2:] for x in json.load(sys.stdin)['acceptedPythonInterpreters']]);"`; \
		if [ ! $$PYTHON_VERSION_IS_CORRECT ]; then echo "Python version $$PYTHON_VERSION is not in acceptedPythonInterpreters"; exit 1; fi; \
	)
	@( \
		rm -rf tests/python/unit/env/; \
		python3 -m venv tests/python/unit/env/; \
		source tests/python/unit/env/bin/activate; \
		pip3 install --upgrade pip; \
		pip3 install --no-cache-dir -r tests/python/unit/requirements.txt; \
		pip3 install --no-cache-dir -r code-env/python/spec/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)/python-lib"; \
		export RESOURCE_FOLDER_PATH="$(PWD)/resource"; \
		pytest tests/python/unit --alluredir=tests/allure_report; \
		deactivate; \
	)
	@echo "[SUCCESS] Running unit tests: Done!"

integration-tests:
	@echo "[START] Running integration tests..."
	@( \
		rm -rf tests/python/integration/env/; \
		python3 -m venv tests/python/integration/env/; \
		source tests/python/integration/env/bin/activate; \
		pip3 install --upgrade pip;\
		pip3 install --no-cache-dir -r tests/python/integration/requirements.txt; \
		pytest tests/python/integration --alluredir=tests/allure_report; \
		deactivate; \
	)
	@echo "[SUCCESS] Running integration tests: Done!"

tests: unit-tests integration-tests

dist-clean:
	rm -rf dist