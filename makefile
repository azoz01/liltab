get_dependencies:
	pip install -r requirements.txt

prepare_code:
	black --line-length=79 liltab
	flake8 liltab
	black --line-length=79 test
	flake8 test

run_tests:
	export PYTHONPATH=`pwd` && pytest