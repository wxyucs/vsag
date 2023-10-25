
add_library (vsag_mockimpl SHARED vsag/simpleflat.cpp vsag/factory.cpp)

add_executable (example_diskann_mockimpl ../examples/cpp/example_diskann.cpp)
target_link_libraries (example_diskann_mockimpl vsag_mockimpl)

add_executable (example_hnsw_mockimpl ../examples/cpp/example_hnsw.cpp)
target_link_libraries (example_hnsw_mockimpl vsag_mockimpl)

install(TARGETS vsag_mockimpl
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

find_package (Git)
add_custom_target (version_mockimpl
  ${CMAKE_COMMAND} -D SRC=${CMAKE_SOURCE_DIR}/mockimpl/vsag/version.h.in
                   -D DST=${CMAKE_SOURCE_DIR}/mockimpl/vsag/version.h
                   -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
                   -P ${CMAKE_SOURCE_DIR}/cmake/GenerateVersionHeader.cmake
  )
add_dependencies (vsag_mockimpl version_mockimpl)

if (ENABLE_TESTS)
    add_executable (tests_mockimpl
            ../tests/test_hnsw.cpp
            ../tests/test_factory.cpp
    )
    target_include_directories(tests_mockimpl PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
    target_link_libraries (tests_mockimpl PRIVATE Catch2::Catch2WithMain vsag_mockimpl)
    add_dependencies(tests_mockimpl spdlog Catch2)
endif ()
