
CMAKE_GENERATOR ?= "Unix Makefiles"
CMAKE_INSTALL_PRECIX ?= "/usr/local/"
VSAG_CMAKE_ARGS = -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DENABLE_TESTS=1 -DENABLE_PYBINDS=1 -G ${CMAKE_GENERATOR} -S. -Bbuild
COMPILE_JOBS ?= 4

debug:
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}

release:
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel ${COMPILE_JOBS}

format:
	find include/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find src/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find python_bindings/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find examples/cpp/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find mockimpl/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find tests/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i

test: debug
	./build/tests -d yes
	./build/mockimpl/tests_mockimpl -d yes

asan:
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}

test_asan: asan
	./build/tests -d yes || true
	./build/mockimpl/tests_mockimpl -d yes || true

cov:
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON -DENABLE_CCACHE=ON
	cmake --build build --parallel ${COMPILE_JOBS}

test_cov: cov
	./build/tests -d yes
	./build/mockimpl/tests_mockimpl -d yes

benchmark:
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel ${COMPILE_JOBS}
	pip3 install -r requirements.txt
	python3 benchs/run.py

clean:
	rm -rf build/*

install: debug
	cmake --install build/

.PHONY: test benchmark clean

