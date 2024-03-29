SHELL:=/bin/bash

# Plugin identifier for the main CPU version of the plugin
# This will be used as prefix for the GPU versions
PLUGIN_ID="timeseries-forecast"

# Variables set automatically
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${PLUGIN_ID}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`

plugin:
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
	@zip --delete dist/${archive_file_name} "tests/*"
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
	@echo "[START] Saving ZIP archive of the plugin (GPU - CUDA ${CUDA_VERSION})..."
	@( \
		plugin_id_gpu="${PLUGIN_ID}-gpu-cuda${CUDA_VERSION}"; \
		echo "Modifying a few files to make the plugin GPU-ready. Fasten your seatbelt."; \
		sed -i "" "s/${PLUGIN_ID}/$${plugin_id_gpu}/g" plugin.json; \
		sed -i "" "s/: \[/: \[\"GPU\",/g" plugin.json; \
		sed -i "" "s/\"label\": \"Forecast (deprecated)\"/\"label\": \"Forecast (GPU - CUDA ${CUDA_VERSION} - deprecated)\"/g" plugin.json; \
		cat plugin.json | json_pp > /dev/null; \
		sed -i "" "s/mxnet.*/mxnet-cu${CUDA_VERSION}==${MXNET_VERSION}/g" code-env/python/spec/requirements.txt; \
		sed -i "" "s/'cpu'/'gpu'/g" custom-recipes/${PLUGIN_ID}-1-train-evaluate/recipe.json; \
		sed -i "" "s/\"defaultValue\": 32/\"defaultValue\": 128/g" custom-recipes/${PLUGIN_ID}-1-train-evaluate/recipe.json; \
		git mv custom-recipes/${PLUGIN_ID}-1-train-evaluate custom-recipes/$${plugin_id_gpu}-1-train-evaluate; \
		git mv custom-recipes/${PLUGIN_ID}-2-predict custom-recipes/$${plugin_id_gpu}-2-predict; \
		sed -i "" "s/cpu_devices/gpu_devices/g" resource/select_gpu_devices.py; \
		sed -i "" "s/{CUDA_VERSION}/${CUDA_VERSION}/g" python-lib/dku_constants.py; \
		git_stash=`git stash create` && echo "Stached modifications to $${git_stash:-HEAD}"; \
		rm -rf dist && mkdir dist; \
		archive_file_name_gpu="dss-plugin-$${plugin_id_gpu}-${plugin_version}.zip"; \
		git archive -v -9 --format zip -o dist/$${archive_file_name_gpu} $${git_stash:-HEAD}; \
		zip --delete dist/$${archive_file_name_gpu} "tests/*"; \
		git reset --hard HEAD; \
		git stash clear; \
	)
	@echo "[SUCCESS] Saving ZIP archive of the plugin (GPU - CUDA ${CUDA_VERSION}): Done!"

unit-tests:
	@echo "Running unit tests..."
	@( \
		set -e; \
		PYTHON_VERSION=`python3 -V 2>&1 | sed 's/[^0-9]*//g' | cut -c 1,2`; \
		PYTHON_VERSION_IS_CORRECT=`cat code-env/python/desc.json | python3 -c "import sys, json; print(str($$PYTHON_VERSION) in [x[-2:] for x in json.load(sys.stdin)['acceptedPythonInterpreters']]);"`; \
		if [ $$PYTHON_VERSION_IS_CORRECT == "False" ]; then echo "Python version $$PYTHON_VERSION is not in acceptedPythonInterpreters"; exit 1; else echo "Python version $$PYTHON_VERSION is in acceptedPythonInterpreters"; fi; \
		rm -rf ./env/; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip;\
		pip install --no-cache-dir -r tests/python/unit/requirements.txt; \
		pip install --no-cache-dir -r code-env/python/spec/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)/python-lib"; \
		export RESOURCE_FOLDER_PATH="$(PWD)/resource"; \
		pytest tests/python/unit --alluredir=tests/allure_report || ret=$$?; exit $$ret \
	)

integration-tests:
	@echo "Running integration tests..."
	@( \
		set -e; \
		rm -rf ./env/; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip;\
		pip install --no-cache-dir -r tests/python/integration/requirements.txt; \
		pytest tests/python/integration --alluredir=tests/allure_report || ret=$$?; exit $$ret \
	)

tests: unit-tests integration-tests

dist-clean:
	rm -rf dist
