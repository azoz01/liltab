get_dependencies:
	pip install -r requirements.txt

prepare_code:
	black --line-length=79 liltab
	flake8 liltab
	black --line-length=79 test
	flake8 test

run_tests:
	export PYTHONPATH=`pwd` && pytest

get_coverage: |
	export PYTHONPATH=`pwd` && pytest -vv --cov=liltab --junitxml=pytest.xml --cov-report=term-missing | tee pytest-coverage.txt
	cat pytest-coverage.txt
	rm pytest-coverage.txt .coverage pytest.xml
