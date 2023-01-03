LINT_PATHS = *.py custom_envs/ optim/


# Type check
type:
	pytype

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

ruff:
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero --line-length 127

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: format type lint

.PHONY: lint type format check-codestyle
