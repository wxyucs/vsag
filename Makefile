
CMAKE_GENERATOR ?= "Unix Makefiles"
VSAG_CMAKE_ARGS = -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_TESTS=1 -DENABLE_PYBINDS=1 -G ${CMAKE_GENERATOR} -S. -Bbuild

debug:
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --parallel 4

release:
	cmake ${VSAG_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel 4

format:
	find include/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i
	find src/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i

test: debug
	./build/tests -d yes

benchmark:
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel 4
	pip3 install -r requirements.txt
	python3 benchs/run.py

clean:
	rm -rf build/*

.PHONY: test benchmark clean

