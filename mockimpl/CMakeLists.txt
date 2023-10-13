
add_library (vsag_mockimpl SHARED vsag/simpleflat.cpp vsag/factory.cpp)

add_executable (example_diskann_mockimpl ../examples/cpp/example_diskann.cpp)
target_link_libraries (example_diskann_mockimpl vsag_mockimpl)

add_executable (example_hnsw_mockimpl ../examples/cpp/example_hnsw.cpp)
target_link_libraries (example_hnsw_mockimpl vsag_mockimpl)

install(TARGETS vsag_mockimpl
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
