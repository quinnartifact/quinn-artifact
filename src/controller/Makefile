# Makefile for QUINN Controller

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. -Ishm
LDFLAGS = -lrt

# Targets
TARGETS = test_shm_reader

.PHONY: all clean test

all: $(TARGETS)

test_shm_reader: test/test_shm_reader.cpp shm/budget_shm.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Python tests
test: test_shm_reader
	@echo "Running Python tests..."
	@echo "======================"
	@echo "Test 1: Allocator"
	@echo "======================"
	python3 test/test_allocator.py --model_dir ../../model/sift100m-learning --model_type gbdt || true
	@echo ""
	@echo "======================"
	@echo "Test 2: Shared Memory"
	@echo "======================"
	python3 test/test_shm.py --num_queries 100
	@echo ""
	@echo "All tests completed!"

clean:
	rm -f $(TARGETS)
	rm -f *.o
	rm -rf __pycache__
	rm -rf test_output

help:
	@echo "QUINN Controller Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all C++ targets"
	@echo "  test_shm_reader  - Build C++ shared memory reader test"
	@echo "  test             - Run all tests (requires Python dependencies)"
	@echo "  clean            - Remove all build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build all targets"
	@echo "  make test               # Run all tests"
	@echo "  make clean              # Clean build artifacts"
