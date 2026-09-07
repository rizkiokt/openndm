# C++ unit tests (NFR-QA-1). Catch2 is fetched rather than vendored so the
# default build stays dependency free; configure with -DOPENNDM_BUILD_TESTS=ON.
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(openndm_tests
  test_matrix.cpp
  test_geometry.cpp
  test_xslib.cpp
  test_kernels.cpp
)
target_link_libraries(openndm_tests PRIVATE openndm_core Catch2::Catch2WithMain)
target_include_directories(openndm_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(openndm_tests)
