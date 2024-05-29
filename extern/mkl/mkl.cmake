# find_package(MKL CONFIG REQUIRED)


if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" AND ENABLE_INTEL_MKL)
    set(POSSIBLE_OMP_PATHS "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so;/usr/lib/x86_64-linux-gnu/libiomp5.so;/opt/intel/lib/intel64_lin/libiomp5.so;/opt/intel/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin/libiomp5.so")
    foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
        if (EXISTS ${POSSIBLE_OMP_PATH})
            get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
        endif()
    endforeach()
    
    if(NOT OMP_PATH)
        message(FATAL_ERROR "Could not find Intel OMP in standard locations")
    endif()
    link_directories(${OMP_PATH})
    
    set(POSSIBLE_MKL_LIB_PATHS "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;/opt/intel/mkl/lib/intel64/libmkl_core.so")
    foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
        if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
            get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
        endif()
    endforeach()
    
    set(POSSIBLE_MKL_INCLUDE_PATHS "/opt/intel/oneapi/mkl/latest/include;/usr/include/mkl;/opt/intel/mkl/include/;")
    foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
        if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH})
            set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
        endif()
    endforeach()
    
    if(NOT MKL_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations")
    endif()
    if(NOT MKL_INCLUDE_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations")
    endif()
    
    if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
    elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
    else()
        message(FATAL_ERROR "Despite finding MKL, libmkl_def.so was not found in expected locations.")
    endif()
    
    link_directories (${MKL_PATH})
    include_directories (${MKL_INCLUDE_PATH})
    set (BLAS_LIBRARIES
        ${MKL_PATH}/libmkl_intel_lp64.so
        ${MKL_PATH}/libmkl_sequential.so
        ${MKL_PATH}/libmkl_core.so
        ${MKL_PATH}/libmkl_def.so
        ${MKL_PATH}/libmkl_avx2.so
        ${MKL_PATH}/libmkl_mc3.so
        ${MKL_PATH}/libmkl_gf_lp64.so
        ${MKL_PATH}/libmkl_core.so
        ${MKL_PATH}/libmkl_intel_thread.so
        iomp5
    )
    message ("enable intel-mkl as blas backend")
else ()
    set (BLAS_LIBRARIES
        libopenblas.a
    )
    message ("enable openblas as blas backend")
endif ()
