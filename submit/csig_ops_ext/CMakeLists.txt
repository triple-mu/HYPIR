cmake_minimum_required(VERSION 3.18)
project(custom_op LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "70-real;70-virtual")
endif ()

# Python_EXECUTABLE 由 setup.py 经 -D 注入；这里不读 env、无相对回退。
if (NOT DEFINED Python_EXECUTABLE OR NOT EXISTS "${Python_EXECUTABLE}")
    message(FATAL_ERROR "Python_EXECUTABLE missing: setup.py must pass -DPython_EXECUTABLE=<absolute>.")
endif ()
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" CACHE STRING "Torch CMake prefix" FORCE)
set(Torch_DIR "${TORCH_CMAKE_PREFIX_PATH}/Torch" CACHE PATH "Torch package directory" FORCE)
set(Caffe2_DIR "${TORCH_CMAKE_PREFIX_PATH}/Caffe2" CACHE PATH "Caffe2 package directory" FORCE)
find_package(Torch REQUIRED)

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/custom_op/csrc/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/custom_op/csrc/*.cu"
)

if (NOT DEFINED EXT_SUFFIX)
    set(EXT_SUFFIX ".so")
endif ()

add_library(_C SHARED ${SOURCES})
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")
# 四个算子全部经 TORCH_LIBRARY 注册（无 pybind11 直绑 Tensor），故不需要链 torch_python。
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} Python::Module)

target_compile_options(_C PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fdiagnostics-color=always>
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math --expt-relaxed-constexpr
        --ftz=true --prec-div=false --prec-sqrt=false
        -Xptxas=-O3
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF2_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -Xptxas=-v
        -Xptxas=--warn-on-spills
        >
)
