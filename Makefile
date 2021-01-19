# Variables set automatically
plugin_id=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['id']).replace('/',''))"`
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`

plugin:
ifndef GPU
	$(error Please set GPU variable to TRUE/FALSE)
endif
ifeq ($(GPU), TRUE)
	@$(MAKE) plugin-gpu
else
	@$(MAKE) plugin-cpu
endif

plugin-cpu:
	@echo "[START] Saving ZIP archive of the plugin (CPU)..."
	@cat plugin.json | json_pp > /dev/null
	@rm -rf dist
	@mkdir dist
	@echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json
	@git archive -v -9 --format zip -o dist/${archive_file_name} HEAD
	@zip -u dist/${archive_file_name} release_info.json
	@rm release_info.json
	@echo "[SUCCESS] Saving ZIP archive of the plugin (CPU): Done!"

plugin-gpu:
ifndef MXNET_VERSION
	$(error Please set MXNET_VERSION variable e.g., 1.7.0)
endif
ifndef CUDA_VERSION
	$(error Please set CUDA_VERSION variable e.g., 102 for cuda 10.2)
endif
	@echo "[START] Saving ZIP archive of the plugin (GPU - mxnet-cu${CUDA_VERSION})..."
	@( \
		plugin_id_gpu="${plugin_id}-gpu-cu${CUDA_VERSION}"; \
		archive_file_name_gpu="dss-plugin-$${plugin_id_gpu}-${plugin_version}.zip"; \
		mxnet_version="mxnet-cu${CUDA_VERSION}"; \
		echo "Modifying a few files to make the plugin GPU-ready. Fasten your seatbelt."; \
		git_stash=`git stash create`; \
		sed -i "" "s/${plugin_id}/$${plugin_id_gpu}/g" plugin.json; \
		echo "Stached modifications to $${git_stash:-HEAD}"; \
		rm -rf dist && mkdir dist; \
		git archive -v -9 --format zip -o dist/$${archive_file_name_gpu} $${git_stash:-HEAD}; \
		git stash drop; \
	)
	@echo "[SUCCESS] Saving ZIP archive of the plugin (GPU - mxnet-cu${CUDA_VERSION}): Done!"

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