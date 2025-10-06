.PHONY: setup install test lint format clean build docker cpp cpp-clean

setup:
	pip install -e .
	pip install -e ".[dev]"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=flashback --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 flashback/ tests/
	mypy flashback/

format:
	black flashback/ tests/
	isort flashback/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

docker:
	docker build -t flashback .

run-example:
	flashback run --config config/backtest.yaml

benchmark:
	python -m flashback.benchmark

docs:
	sphinx-build -b html docs/ docs/_build/html

# C++ extension build targets
cpp:
	@echo "Building C++ matching engine extension..."
	@mkdir -p cpp/build
	@cd cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release
	@cd cpp/build && make -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo "C++ extension built successfully: flashback/market/_match_cpp.so"

cpp-clean:
	@echo "Cleaning C++ build artifacts..."
	@rm -rf cpp/build
	@rm -f flashback/market/_match_cpp.so
	@echo "C++ build artifacts cleaned"

cpp-test:
	@echo "Testing C++ extension..."
	@python -c "import flashback.market._match_cpp as cpp; print('C++ extension loaded successfully')"

bench:
	@echo "Running performance benchmarks..."
	@python flashback/utils/bench_cpp.py --orders 1000000 --trials 3
	@echo "Generating benchmark plots..."
	@python flashback/metrics/bench_plots.py bench_results.csv

ci-bench:
	@echo "Running CI benchmark process..."
	@./scripts/ci_benchmark.sh

