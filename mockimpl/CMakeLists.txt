
add_library (vsag_mockimpl SHARED simpleflat.cpp)

add_executable (example_simpleflat example_simpleflat.cpp)
target_link_libraries (example_simpleflat vsag_mockimpl)
