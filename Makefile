
debug:
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug
	cmake --build build

format:
	find include/ -iname *.h -o -iname *.cpp | xargs clang-format -i
	find src/ -iname *.h -o -iname *.cpp | xargs clang-format -i

test: debug
	./build/tests

clean:
	rm -rf build/*

.PHONY: always clean

