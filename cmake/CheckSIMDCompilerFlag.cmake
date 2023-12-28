
file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp "#include <immintrin.h>\nint main() { __m512 a, b; a = _mm512_sub_ps(a, b); return 0; }")
try_compile(COMPILER_AVX512_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512
    ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp
    COMPILE_DEFINITIONS "-mavx512f"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp "#include <immintrin.h>\nint main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }")
try_compile(COMPILER_AVX2_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx2
    ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp
    COMPILE_DEFINITIONS "-mavx2 -mfma"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp "#include <immintrin.h>\nint main() { __m256 a, b; a = _mm256_sub_ps(a, b); return 0; }")
try_compile(COMPILER_AVX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx
    ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp
    COMPILE_DEFINITIONS "-mavx"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp "#include <immintrin.h>\nint main() { __m128 a, b; a = _mm_sub_ps(a, b); return 0; }")
try_compile(COMPILER_SSE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sse
    ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp
    COMPILE_DEFINITIONS "-msse"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp "#include <immintrin.h>\nint main() { __m512 a, b; a = _mm512_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_AVX512_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512
    ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp "#include <immintrin.h>\nint main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }")
try_compile(RUNTIME_AVX2_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx2
    ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp "#include <immintrin.h>\nint main() { __m256 a, b; a = _mm256_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_AVX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx
    ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp "#include <immintrin.h>\nint main() { __m128 a, b; a = _mm_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_SSE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sse
    ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )
