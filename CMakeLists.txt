cmake_minimum_required(VERSION 3.10)
set(name_project "main")

project(main LANGUAGES C CXX CUDA)
# 查找CUDA
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    unset (CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
    option (CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    set(CMAKE_CUDA_ARCHITECTURES "120a")
    set(SYS_CUDART_PATH "/usr/local/cuda")
    include_directories (${SYS_CUDART_PATH}/include)
    link_directories (${SYS_CUDART_PATH}/lib64)
    set(CUDA_NVCC_FLAGS -gencode arch=compute_120a,code=sm_120a)
    file(GLOB mainfile "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSC_DISABLE_API_VERSION_CHECK")
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    cuda_add_executable(${name_project} ${mainfile})
    target_link_libraries(${name_project}
        ${CUDART_LIBRARY}
        ${CUBLASLT_LIBRARY}
    )
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSC_DISABLE_API_VERSION_CHECK")
    file(GLOB mainfile "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    add_executable(${name_project} ${mainfile})
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pg -march=native -mtune=native -fpermissive") 

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/external/systemc/include")
file(GLOB_RECURSE SYSTEMC_LIB "${CMAKE_CURRENT_SOURCE_DIR}/external/systemc/lib/*.so")
target_link_libraries(${name_project} ${SYSTEMC_LIB})

target_link_libraries(${name_project} stdc++fs)
