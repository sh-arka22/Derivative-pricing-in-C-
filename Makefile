CXX       := g++
CXXFLAGS  := -std=c++17 -Wall -Wextra -Wpedantic
OPTFLAGS  := -O3 -march=native
DBGFLAGS  := -g -O0 -fsanitize=address
INCLUDES  := -I include
SRCDIR    := src
BUILDDIR  := build

LIBSRC    := $(SRCDIR)/quantpricer.cpp

# Targets
.PHONY: all demo test examples clean release debug dashboard viz paper-trade download-data generate-data backtest smoke-test

all: release

release: CXXFLAGS += $(OPTFLAGS)
release: demo test examples

debug: CXXFLAGS += $(DBGFLAGS)
debug: demo test

demo: $(BUILDDIR)/quantpricer_demo
test: $(BUILDDIR)/test_runner
examples: $(BUILDDIR)/mc_european $(BUILDDIR)/mc_exotic $(BUILDDIR)/fdm_solver \
          $(BUILDDIR)/greeks_engine $(BUILDDIR)/vol_surface \
          $(BUILDDIR)/heston_pricer $(BUILDDIR)/jump_diffusion \
          $(BUILDDIR)/american_options $(BUILDDIR)/barrier_options \
          $(BUILDDIR)/multi_asset_options $(BUILDDIR)/risk_management \
          $(BUILDDIR)/fixed_income $(BUILDDIR)/rate_models \
          $(BUILDDIR)/orderbook_demo

$(BUILDDIR)/%: examples/%.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

$(BUILDDIR)/quantpricer_demo: examples/main_demo.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

$(BUILDDIR)/test_runner: tests/test_runner.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run-demo: demo
	./$(BUILDDIR)/quantpricer_demo

run-tests: test
	./$(BUILDDIR)/test_runner

$(BUILDDIR)/generate_data: tools/generate_data.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

$(BUILDDIR)/generate_showcase: tools/generate_showcase.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

showcase: release $(BUILDDIR)/generate_showcase
	./$(BUILDDIR)/generate_showcase
	python3 tools/render_showcase.py

$(BUILDDIR)/pricer_service: tools/pricer_service.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

dashboard: release $(BUILDDIR)/pricer_service
	cd dashboard && streamlit run app.py

viz: release $(BUILDDIR)/generate_data
	./$(BUILDDIR)/generate_data
	python3 tools/visualize.py

paper-trade: build/paper_trader
	./build/paper_trader config/paper_trading.json

build/paper_trader: src/paper_trader.cpp include/trading/*.h
	@mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev && make paper_trader

# Download real OHLCV data from Yahoo Finance (primary)
download-data:
	python3 tools/download_data.py

# Generate synthetic OHLCV data via GBM (fallback if offline)
generate-data: build/generate_sample_data
	./build/generate_sample_data

build/generate_sample_data: tools/generate_sample_data.cpp
	@mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev && make generate_sample_data

# Run backtest comparison across all strategies (Day 13)
backtest: build/run_backtest
	./build/run_backtest config/paper_trading.json

build/run_backtest: tools/run_backtest.cpp
	@mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev && make run_backtest

# Full smoke test (Day 14)
smoke-test: generate-data
	bash tools/smoke_test.sh

clean:
	rm -rf $(BUILDDIR) *.csv plots/
