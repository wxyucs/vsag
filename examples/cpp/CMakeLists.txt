add_executable (example_hnsw example_hnsw.cpp)
target_link_libraries (example_hnsw vsag)

add_executable (example_kmeans example_kmeans.cpp)
target_link_libraries (example_kmeans vsag)

add_executable (example_diskann example_diskann.cpp)
target_link_libraries (example_diskann vsag_static)

add_executable (simple_hnsw simple_hnsw.cpp)
target_link_libraries (simple_hnsw vsag_static)
