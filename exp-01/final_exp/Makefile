.PHONY: test quick all report clean
test:
	python -m pytest -q
quick:
	python experiments/run_all.py --quick
	python experiments/summarize.py
all:
	python experiments/run_all.py
	python experiments/summarize.py
report:
	python experiments/summarize.py
clean:
	rm -rf results/*