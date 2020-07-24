# Public variable to be set by the user in the Makefile
TARGET_DSS_VERSION=8.0

# Private variables to be set by the user in the environment
ifndef DKU_PLUGIN_DEVELOPER_ORG
$(error the DKU_PLUGIN_DEVELOPER_ORG environment variable is not set)
endif
ifndef DKU_PLUGIN_DEVELOPER_TOKEN
$(error the DKU_PLUGIN_DEVELOPER_TOKEN environment variable is not set)
endif
ifndef DKU_PLUGIN_DEVELOPER_REPO_URL
$(error the DKU_PLUGIN_DEVELOPER_REPO_URL environment variable is not set)
endif

# evaluate additional variable
plugin_id=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['id']).replace('/',''))"`
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${plugin_version}.zip"
artifact_repo_target="${DKU_PLUGIN_DEVELOPER_REPO_URL}/${TARGET_DSS_VERSION}/${DKU_PLUGIN_DEVELOPER_ORG}/${plugin_id}/${plugin_version}/${archive_file_name}"
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

submit: plugin
	@echo "[START] Publishing archive to artifact repository..."
	@curl -H "Authorization: Bearer ${DKU_PLUGIN_DEVELOPER_TOKEN}>" -X PUT ${artifact_repo_target} -T dist/${archive_file_name}
	@echo "[SUCCESS] Publishing archive to artifact repository: Done!"

unit-tests:
	@echo "[START] Running unit tests..."
	@( \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip install --no-cache-dir -r tests/python/requirements.txt; \
		pip install --no-cache-dir -r code-env/python/spec/requirements.txt; \
		python3 -m pytest tests/python/unit/; \
		source deactivate; \
	)
	@echo "[SUCCESS] Running unit tests: Done!"

integration-tests:
	@echo "[START] Running integration tests..."
	@# TODO add integration tests in v2
	@echo "No integration tests detected :'("
	@echo "[SUCCESS] Running integration tests: Done!"

tests: unit-tests integration-tests

dist-clean:
	rm -rf dist
