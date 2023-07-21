
debug:
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --parallel 4

release:
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel 4

format:
	find include/ -iname *.h -o -iname *.cpp | xargs clang-format -i
	find src/ -iname *.h -o -iname *.cpp | xargs clang-format -i

test: debug
	./build/tests -d yes

benchmark: release
	./build/bench_random1m

clean:
	rm -rf build/*

.PHONY: test benchmark clean

