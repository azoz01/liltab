get_dependencies:
	pip install -r requirements.txt

prepare_code:
	black --line-length=79 liltab
	flake8 liltab

run_tests:
	export PYTHONPATH=`pwd`/liltab && pytest