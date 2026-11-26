.PHONY: test all report clean
test:
	python -m pytest -q
all:
	python experiments/run_all.py
report:
	python experiments/phase7_report.py
clean:
	rm -rf results/* docs/*.md __pycache__ .pytest_cache