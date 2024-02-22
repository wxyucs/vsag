
CMAKE_GENERATOR ?= "Unix Makefiles"
CMAKE_INSTALL_PRECIX ?= "/usr/local/"
VSAG_CMAKE_ARGS = -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DENABLE_TESTS=1 -DENABLE_PYBINDS=1 -G ${CMAKE_GENERATOR} -S. -Bbuild
COMPILE_JOBS ?= 4

.PHONY: help
help:                   ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: debug
debug:                  ## Build vsag with debug options.
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}

.PHONY: release
release:                ## Build vsag with release options.
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel ${COMPILE_JOBS}

.PHONY: fmt
fmt:                    ## Format codes.
	find include/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find src/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find python_bindings/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find examples/cpp/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find mockimpl/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find tests/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i

.PHONY: test
test:                   ## Build and run unit tests.
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}
	./build/tests -d yes
	./build/mockimpl/tests_mockimpl -d yes

.PHONY: test_asan
test_asan:              ## Build and run unit tests with AddressSanitizer option.
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}
	./build/tests -d yes || true
	./build/mockimpl/tests_mockimpl -d yes || true

.PHONY: test_cov
test_cov:               ## Build and run unit tests with code coverage enabled.
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}
	./build/tests -d yes
	./build/mockimpl/tests_mockimpl -d yes

.PHONY: benchmark
benchmark:              ## Run benchmarks.
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel ${COMPILE_JOBS}
	pip3 install -r docker/requirements.txt
	python3 benchs/run.py

.PHONY: clean
clean:                  ## Clear build/ directory.
	rm -rf build/*

.PHONY: install
install: debug          ## Build and install the debug version of vsag.
	cmake --install build/

