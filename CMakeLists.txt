cmake_minimum_required(VERSION 3.16)
project(openndm VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Open Nodal Diffusion Method")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(OPENNDM_BUILD_PYTHON "Build the pybind11 extension module" ON)
option(OPENNDM_BUILD_TESTS "Build the C++ unit tests" OFF)
option(OPENNDM_USE_OPENMP "Enable OpenMP shared-memory parallelism" ON)

# ---------------------------------------------------------------- core library
add_library(openndm_core
  src/geometry.cpp
  src/xslib.cpp
  src/matrix.cpp
  src/cmfd.cpp
  src/kernel.cpp
  src/kernel_fdm.cpp
  src/kernel_nem.cpp
  src/kernel_sanm.cpp
  src/solver.cpp
)
target_include_directories(openndm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(openndm_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(openndm_core PRIVATE /W4)
else()
  target_compile_options(openndm_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(OPENNDM_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(openndm_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenNDM: OpenMP enabled")
  else()
    message(STATUS "OpenNDM: OpenMP not found, building single-threaded")
  endif()
endif()

# ------------------------------------------------------------- python bindings
if(OPENNDM_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core src/bindings.cpp)
  target_link_libraries(_core PRIVATE openndm_core)
  target_compile_definitions(_core PRIVATE
    OPENNDM_VERSION="${PROJECT_VERSION}")
  install(TARGETS _core LIBRARY DESTINATION openndm)
endif()

# ----------------------------------------------------------------------- tests
if(OPENNDM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
