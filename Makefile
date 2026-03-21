CXX       := g++
CXXFLAGS  := -std=c++17 -Wall -Wextra -Wpedantic
OPTFLAGS  := -O3 -march=native
DBGFLAGS  := -g -O0 -fsanitize=address
INCLUDES  := -I include
SRCDIR    := src
BUILDDIR  := build

LIBSRC    := $(SRCDIR)/quantpricer.cpp

# Targets
.PHONY: all demo test examples clean release debug dashboard viz

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

$(BUILDDIR)/pricer_service: tools/pricer_service.cpp $(LIBSRC) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) $(LIBSRC) $< -o $@ -lm

dashboard: release $(BUILDDIR)/pricer_service
	cd dashboard && streamlit run app.py

viz: release $(BUILDDIR)/generate_data
	./$(BUILDDIR)/generate_data
	python3 tools/visualize.py

clean:
	rm -rf $(BUILDDIR) *.csv plots/
