add_executable (sq8_per_vector_benchmark sq8_per_vector.cpp)
target_link_libraries (sq8_per_vector_benchmark PRIVATE vsag)

add_executable (sq8_adc_reference sq8_adc_reference.cpp)
target_link_libraries (sq8_adc_reference PRIVATE vsag)
