cmake_minimum_required(VERSION 3.20.0)

project(mlir-tutorial LANGUAGES CXX C)

# local specific setting
# here are paths to my llvm mlir builds

set(MLIR_DIR "/home/dran/Projects/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR build")
set(LLVM_DIR "/home/dran/Projects/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVM build")

# Force the C/C++ compilers to use the LLVM linker (lld)
add_link_options("-fuse-ld=lld")

# Match LLVM's build settings by disabling RTTI and exceptions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

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

# adding our pass
add_library(CountAffineForPass
        CountAffineFor.cpp
)

# Link the core MLIR components our pass needs to function.
target_link_libraries(CountAffineForPass
        PRIVATE
        MLIRIR
        MLIRPass
        MLIRFuncDialect
        MLIRAffineDialect # Crucial for matching and analyzing affine.for loops
)

# add our executable
add_executable(count-opt
        count-opt.cpp
)

# Link your custom pass AND the MLIR opt utilities
target_link_libraries(count-opt
        PRIVATE
        CountAffineForPass    # our custom pass
        MLIROptLib            # Provides the core infrastructure for mlir-opt clones
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRFuncDialect
        MLIRAffineDialect
)

# adding the other pass
add_library(MutateAffineForPass
        MutateAffineFor.cpp
)

# Link the core MLIR components our pass needs to function.
target_link_libraries(MutateAffineForPass
        PRIVATE
        MLIRIR
        MLIRPass
        MLIRFuncDialect
        MLIRAffineDialect # Crucial for matching and analyzing affine.for loops
        MLIRSCFDialect
        MLIRArithDialect
)

# add our executable
add_executable(mutate-opt
        mutate-opt.cpp
)

# Link your custom pass AND the MLIR opt utilities
target_link_libraries(mutate-opt
        PRIVATE
        MutateAffineForPass    # our custom pass
        MLIROptLib            # Provides the core infrastructure for mlir-opt clones
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRFuncDialect
        MLIRAffineDialect
        MLIRSCFDialect
        MLIRArithDialect
)