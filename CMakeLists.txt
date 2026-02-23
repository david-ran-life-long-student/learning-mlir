cmake_minimum_required(VERSION 3.20.0)

project(mlir-tutorial LANGUAGES CXX C)

# local specific setting
# here are paths to my llvm mlir builds

set(MLIR_DIR "/home/dran/Projects/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR build")
set(LLVM_DIR "/home/dran/Projects/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVM build")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/externals/llvm-project)
include_directories(${PROJECT_BINARY_DIR})
