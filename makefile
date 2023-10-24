get_dependencies:
	pip install -r requirements.txt

prepare_code:
	black --line-length=100 liltab
	flake8 liltab --max-line-length=100
	black --line-length=100 test
	flake8 test --max-line-length=100
	black --line-length=100 bin
	flake8 bin --max-line-length=100

run_tests:
	export PYTHONPATH=`pwd` && pytest

get_coverage: |
	export PYTHONPATH=`pwd` && pytest -vv --cov=liltab --cov-report=term-missing
	rm .coverage

release: |
	python -m build
	twine upload dist/*